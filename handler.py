import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO
import websocket
import uuid
import tempfile
import socket
import traceback

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 500
# Websocket reconnection behaviour (can be overridden through environment variables)
# NOTE: more attempts and diagnostics improve debuggability whenever ComfyUI crashes mid-job.
#   • WEBSOCKET_RECONNECT_ATTEMPTS sets how many times we will try to reconnect.
#   • WEBSOCKET_RECONNECT_DELAY_S sets the sleep in seconds between attempts.
#
# If the respective env-vars are not supplied we fall back to sensible defaults ("5" and "3").
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", 5))
WEBSOCKET_RECONNECT_DELAY_S = int(os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", 3))

# Extra verbose websocket trace logs (set WEBSOCKET_TRACE=true to enable)
if os.environ.get("WEBSOCKET_TRACE", "false").lower() == "true":
    # This prints low-level frame information to stdout which is invaluable for diagnosing
    # protocol errors but can be noisy in production – therefore gated behind an env-var.
    websocket.enableTrace(True)

# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Helper: quick reachability probe of ComfyUI HTTP endpoint (port 8188)
# ---------------------------------------------------------------------------


def _comfy_server_status():
    """Return a dictionary with basic reachability info for the ComfyUI HTTP server."""
    try:
        resp = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {
            "reachable": resp.status_code == 200,
            "status_code": resp.status_code,
        }
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _attempt_websocket_reconnect(ws_url, max_attempts, delay_s, initial_error):
    """
    Attempts to reconnect to the WebSocket server after a disconnect.

    Args:
        ws_url (str): The WebSocket URL (including client_id).
        max_attempts (int): Maximum number of reconnection attempts.
        delay_s (int): The delay in seconds between attempts.
        initial_error (Exception): The error that triggered the reconnect attempt.

    Returns:
        websocket.WebSocket: The newly connected WebSocket object.

    Raises:
        websocket.WebSocketConnectionClosedException: If reconnection fails after all attempts.
    """
    print(
        f"worker-comfyui - Websocket connection closed unexpectedly: {initial_error}. Attempting to reconnect..."
    )
    last_reconnect_error = initial_error
    for attempt in range(max_attempts):
        # Log current server status before each reconnect attempt so that we can
        # see whether ComfyUI is still alive (HTTP port 8188 responding) even if
        # the websocket dropped. This is extremely useful to differentiate
        # between a network glitch and an outright ComfyUI crash/OOM-kill.
        srv_status = _comfy_server_status()
        if not srv_status["reachable"]:
            # If ComfyUI itself is down there is no point in retrying the websocket –
            # bail out immediately so the caller gets a clear "ComfyUI crashed" error.
            print(
                f"worker-comfyui - ComfyUI HTTP unreachable – aborting websocket reconnect: {srv_status.get('error', 'status '+str(srv_status.get('status_code')))}"
            )
            raise websocket.WebSocketConnectionClosedException(
                "ComfyUI HTTP unreachable during websocket reconnect"
            )

        # Otherwise we proceed with reconnect attempts while server is up
        print(
            f"worker-comfyui - Reconnect attempt {attempt + 1}/{max_attempts}... (ComfyUI HTTP reachable, status {srv_status.get('status_code')})"
        )
        try:
            # Need to create a new socket object for reconnect
            new_ws = websocket.WebSocket()
            new_ws.connect(ws_url, timeout=10)  # Use existing ws_url
            print(f"worker-comfyui - Websocket reconnected successfully.")
            return new_ws  # Return the new connected socket
        except (
            websocket.WebSocketException,
            ConnectionRefusedError,
            socket.timeout,
            OSError,
        ) as reconn_err:
            last_reconnect_error = reconn_err
            print(
                f"worker-comfyui - Reconnect attempt {attempt + 1} failed: {reconn_err}"
            )
            if attempt < max_attempts - 1:
                print(
                    f"worker-comfyui - Waiting {delay_s} seconds before next attempt..."
                )
                time.sleep(delay_s)
            else:
                print(f"worker-comfyui - Max reconnection attempts reached.")

    # If loop completes without returning, raise an exception
    print("worker-comfyui - Failed to reconnect websocket after connection closed.")
    raise websocket.WebSocketConnectionClosedException(
        f"Connection closed and failed to reconnect. Last error: {last_reconnect_error}"
    )


def validate_input(job_input):
    """
    Validates the input for the handler function.

    Args:
        job_input (dict): The input data to validate.

    Returns:
        tuple: A tuple containing the validated data and an error message, if any.
               The structure is (validated_data, error_message).
    """
    # Validate if job_input is provided
    if job_input is None:
        return None, "Please provide input"

    # Check if input is a string and try to parse it as JSON
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Validate 'workflow' in input
    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    # Validate 'images' in input, if provided
    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return (
                None,
                "'images' must be a list of objects with 'name' and 'image' keys",
            )

    # Optional: API key for Comfy.org API Nodes, passed per-request
    comfy_org_api_key = job_input.get("comfy_org_api_key")

    # Return validated data and no error
    return {
        "workflow": workflow,
        "images": images,
        "comfy_org_api_key": comfy_org_api_key,
    }, None


def check_server(url, retries=500, delay=50):
    """
    Check if a server is reachable via HTTP GET request

    Args:
    - url (str): The URL to check
    - retries (int, optional): The number of times to attempt connecting to the server. Default is 50
    - delay (int, optional): The time in milliseconds to wait between retries. Default is 500

    Returns:
    bool: True if the server is reachable within the given number of retries, otherwise False
    """

    print(f"worker-comfyui - Checking API server at {url}...")
    for i in range(retries):
        try:
            response = requests.get(url, timeout=5)

            # If the response status code is 200, the server is up and running
            if response.status_code == 200:
                print(f"worker-comfyui - API is reachable")
                return True
        except requests.Timeout:
            pass
        except requests.RequestException as e:
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay / 1000)

    print(
        f"worker-comfyui - Failed to connect to server at {url} after {retries} attempts."
    )
    return False


def upload_images(images):
    """
    Upload a list of base64 encoded images to the ComfyUI server using the /upload/image endpoint.

    Args:
        images (list): A list of dictionaries, each containing the 'name' of the image and the 'image' as a base64 encoded string.

    Returns:
        dict: A dictionary indicating success or error.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []

    print(f"worker-comfyui - Uploading {len(images)} image(s)...")

    for image in images:
        try:
            name = image["name"]
            image_data_uri = image["image"]  # Get the full string (might have prefix)

            # --- Strip Data URI prefix if present ---
            if "," in image_data_uri:
                # Find the comma and take everything after it
                base64_data = image_data_uri.split(",", 1)[1]
            else:
                # Assume it's already pure base64
                base64_data = image_data_uri
            # --- End strip ---

            blob = base64.b64decode(base64_data)  # Decode the cleaned data

            # Prepare the form data
            files = {
                "image": (name, BytesIO(blob), "image/png"),
                "overwrite": (None, "true"),
            }

            # POST request to upload the image
            response = requests.post(
                f"http://{COMFY_HOST}/upload/image", files=files, timeout=30
            )
            response.raise_for_status()

            responses.append(f"Successfully uploaded {name}")
            print(f"worker-comfyui - Successfully uploaded {name}")

        except base64.binascii.Error as e:
            error_msg = f"Error decoding base64 for {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.Timeout:
            error_msg = f"Timeout uploading {image.get('name', 'unknown')}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.RequestException as e:
            error_msg = f"Error uploading {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except Exception as e:
            error_msg = (
                f"Unexpected error uploading {image.get('name', 'unknown')}: {e}"
            )
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)

    if upload_errors:
        print(f"worker-comfyui - image(s) upload finished with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }

    print(f"worker-comfyui - image(s) upload complete")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }


def get_available_models():
    """
    Get list of available models from ComfyUI

    Returns:
        dict: Dictionary containing available models by type
    """
    try:
        response = requests.get(f"http://{COMFY_HOST}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()

        # Extract available checkpoints from CheckpointLoaderSimple
        available_models = {}
        if "CheckpointLoaderSimple" in object_info:
            checkpoint_info = object_info["CheckpointLoaderSimple"]
            if "input" in checkpoint_info and "required" in checkpoint_info["input"]:
                ckpt_options = checkpoint_info["input"]["required"].get("ckpt_name")
                if ckpt_options and len(ckpt_options) > 0:
                    available_models["checkpoints"] = (
                        ckpt_options[0] if isinstance(ckpt_options[0], list) else []
                    )

        return available_models
    except Exception as e:
        print(f"worker-comfyui - Warning: Could not fetch available models: {e}")
        return {}


def queue_workflow(workflow, client_id, comfy_org_api_key=None):
    """
    Queue a workflow to be processed by ComfyUI

    Args:
        workflow (dict): A dictionary containing the workflow to be processed
        client_id (str): The client ID for the websocket connection
        comfy_org_api_key (str, optional): Comfy.org API key for API Nodes

    Returns:
        dict: The JSON response from ComfyUI after processing the workflow

    Raises:
        ValueError: If the workflow validation fails with detailed error information
    """
    # Include client_id in the prompt payload
    payload = {"prompt": workflow, "client_id": client_id}

    # Optionally inject Comfy.org API key for API Nodes.
    # Precedence: per-request key (argument) overrides environment variable.
    # Note: We use our consistent naming (comfy_org_api_key) but transform to
    # ComfyUI's expected format (api_key_comfy_org) when sending.
    key_from_env = os.environ.get("COMFY_ORG_API_KEY")
    effective_key = comfy_org_api_key if comfy_org_api_key else key_from_env
    if effective_key:
        payload["extra_data"] = {"api_key_comfy_org": effective_key}
    data = json.dumps(payload).encode("utf-8")

    # Use requests for consistency and timeout
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"http://{COMFY_HOST}/prompt", data=data, headers=headers, timeout=30
    )

    # Handle validation errors with detailed information
    if response.status_code == 400:
        try:
            error_data = response.json()
            if "error" in error_data:
                # Format detailed error message
                error_msg = f"Workflow validation failed: {error_data['error']}"
                if "node_errors" in error_data:
                    for err in error_data["node_errors"]:
                        node_id = err.get("node_id", "unknown")
                        errors = err.get("errors", [])
                        for e in errors:
                            error_msg += f"\n- Node {node_id}: {e.get('message', 'Unknown error')}"
                            if "details" in e:
                                error_msg += f" (Details: {e['details']})"
                raise ValueError(error_msg)
        except json.JSONDecodeError:
            raise ValueError(f"Workflow validation failed: {response.text}")

    response.raise_for_status()
    return response.json()


def get_history(prompt_id):
    """
    Retrieve the history of a prompt from ComfyUI using its prompt ID

    Args:
        prompt_id (str): The ID of the prompt to retrieve history for

    Returns:
        dict: The history of the prompt if found, otherwise an empty dictionary
    """
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}")
    try:
        return response.json().get(prompt_id, {})
    except json.JSONDecodeError:
        print(f"worker-comfyui - Error decoding history for prompt {prompt_id}")
        return {}


def get_images(prompt_id, history=None):
    """
    Fetch images generated by ComfyUI for a given prompt ID

    Args:
        prompt_id (str): The prompt ID to fetch images for
        history (dict, optional): The history of the prompt, if already fetched

    Returns:
        list: A list of image data in bytes
    """
    if history is None:
        history = get_history(prompt_id)

    output_images = []

    for node_id, node_output in history.get("outputs", {}).items():
        if "images" in node_output:
            for image_info in node_output["images"]:
                image_path = (
                    f"http://{COMFY_HOST}/view?filename={image_info['filename']}"
                    f"&type={image_info['type']}&subfolder={image_info.get('subfolder', '')}"
                )
                response = requests.get(image_path)
                output_images.append(response.content)  # Append raw bytes

    return output_images


def get_images_with_metadata(prompt_id, history=None):
    """
    Fetch images generated by ComfyUI for a given prompt ID, including metadata

    Args:
        prompt_id (str): The prompt ID to fetch images for
        history (dict, optional): The history of the prompt, if already fetched

    Returns:
        list: A list of dictionaries containing image data and metadata
    """
    if history is None:
        history = get_history(prompt_id)

    output_images = []

    for node_id, node_output in history.get("outputs", {}).items():
        if "images" in node_output:
            for image_info in node_output["images"]:
                image_path = (
                    f"http://{COMFY_HOST}/view?filename={image_info['filename']}"
                    f"&type={image_info['type']}&subfolder={image_info.get('subfolder', '')}"
                )
                # Add extra params to force no-cache
                image_path += f"&rand={time.time()}"
                response = requests.get(image_path)
                response.raise_for_status()

                image_data = {
                    "data": response.content,  # Raw bytes
                    "filename": image_info["filename"],
                    "type": image_info["type"],
                    "subfolder": image_info.get("subfolder", ""),
                    "node_id": node_id,
                }
                output_images.append(image_data)

    return output_images


def upload_to_s3(images):
    """
    Upload a list of images to an S3-compatible storage and return their URLs

    Args:
        images (list): List of image bytes to upload

    Returns:
        list: List of URLs of the uploaded images
    """
    urls = []

    for idx, image in enumerate(images):
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(image)
            tmp.flush()
            url = rp_upload.upload_image(str(idx), tmp.name)
            urls.append(url)

    return urls


def upload_images_to_s3(images_with_metadata):
    """
    Upload images with metadata to S3 and return URLs with metadata

    Args:
        images_with_metadata (list): List of dicts containing image data and metadata

    Returns:
        list: List of dicts with URLs and metadata
    """
    results = []

    for image_data in images_with_metadata:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(image_data["data"])
            tmp.flush()
            # Use original filename for upload
            url = rp_upload.upload_image(image_data["filename"], tmp.name)
            results.append(
                {
                    "url": url,
                    "filename": image_data["filename"],
                    "type": image_data["type"],
                    "subfolder": image_data["subfolder"],
                    "node_id": image_data["node_id"],
                }
            )
        os.unlink(tmp.name)  # Clean up temp file

    return results


def execute_workflow(workflow, client_id, comfy_org_api_key=None):
    """
    Execute the workflow and monitor progress via WebSocket

    Args:
        workflow (dict): The workflow to execute
        client_id (str): Client ID for WebSocket connection
        comfy_org_api_key (str, optional): Comfy.org API key

    Returns:
        tuple: (prompt_id, history)
    """
    prompt_response = queue_workflow(workflow, client_id, comfy_org_api_key)
    prompt_id = prompt_response["prompt_id"]

    # Create the WebSocket URL with clientId
    ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"

    # Connect to the WebSocket
    ws = websocket.WebSocket()
    ws.connect(ws_url, timeout=10)

    try:
        while True:
            try:
                message = ws.recv()
                if message:
                    # Process message
                    decoded_message = json.loads(message)

                    if decoded_message["type"] == "executing":
                        data = decoded_message["data"]
                        if data["prompt_id"] == prompt_id:
                            if data["node"] is None:
                                # Execution complete
                                break

            except websocket.WebSocketConnectionClosedException as ws_closed:
                # Attempt to reconnect and continue
                ws = _attempt_websocket_reconnect(
                    ws_url,
                    WEBSOCKET_RECONNECT_ATTEMPTS,
                    WEBSOCKET_RECONNECT_DELAY_S,
                    ws_closed,
                )
                # After successful reconnect, continue the loop

            except websocket.WebSocketTimeoutException as ws_timeout:
                print(f"worker-comfyui - WebSocket timeout: {ws_timeout}")
                # Attempt reconnect on timeout as well
                ws = _attempt_websocket_reconnect(
                    ws_url,
                    WEBSOCKET_RECONNECT_ATTEMPTS,
                    WEBSOCKET_RECONNECT_DELAY_S,
                    ws_timeout,
                )

            except Exception as e:
                print(f"worker-comfyui - Unexpected error in WebSocket loop: {e}")
                raise

    finally:
        # Always close the WebSocket connection
        ws.close()

    history = get_history(prompt_id)
    return prompt_id, history


def handler(job):
    """
    Main handler function for processing jobs

    Args:
        job (dict): The job data containing input

    Returns:
        dict: The result of the job processing
    """
    try:
        job_input = job["input"]
        validated_data, error = validate_input(job_input)

        if error:
            return {"error": error}

        workflow = validated_data["workflow"]
        images = validated_data.get("images")
        comfy_org_api_key = validated_data.get("comfy_org_api_key")

        # Check if ComfyUI API is available
        if not check_server(
            f"http://{COMFY_HOST}/object_info",
            COMFY_API_AVAILABLE_MAX_RETRIES,
            COMFY_API_AVAILABLE_INTERVAL_MS,
        ):
            return {"error": "ComfyUI API is not available"}

        # Upload images if provided
        if images:
            upload_result = upload_images(images)
            if upload_result["status"] == "error":
                return {"error": upload_result["message"], "details": upload_result["details"]}

        # Generate a unique client_id for this job
        client_id = str(uuid.uuid4())

        # Execute the workflow
        prompt_id, history = execute_workflow(workflow, client_id, comfy_org_api_key)

        # Get images with metadata
        images_with_metadata = get_images_with_metadata(prompt_id, history)

        # Upload to S3 and get URLs with metadata
        uploaded_images = upload_images_to_s3(images_with_metadata)

        # Optional: include execution metadata
        metadata = {
            "prompt_id": prompt_id,
            "execution_time": history.get("execution_time", "unknown"),
            "nodes_executed": list(history.get("outputs", {}).keys()),
        }

        result = {
            "images": uploaded_images,
            "metadata": metadata,
        }

        if REFRESH_WORKER:
            result["refresh_worker"] = True

        return result

    except ValueError as ve:
        # Handle validation errors
        return {"error": str(ve)}
    except Exception as e:
        # Log full traceback for debugging
        traceback.print_exc()
        return {"error": f"Job failed: {str(e)}"}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
