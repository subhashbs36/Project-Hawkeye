import json
from collections.abc import Callable
from typing import cast, Callable, Tuple
import uuid
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import requests

from anthropic import APIResponse
from anthropic.types import ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam, BetaUsage

from agent.llm_utils.oaiclient import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
from agent.llm_utils.utils import is_image_path, encode_image
import time
import re

OUTPUT_DIR = "./tmp/outputs"

def extract_data(input_string, data_type):
    # Regular expression to extract content starting from '```python' until the end if there are no closing backticks
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    # Extract content
    # re.DOTALL allows '.' to match newlines as well
    matches = re.findall(pattern, input_string, re.DOTALL)
    # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
    return matches[0][0].strip() if matches else input_string

class VLMAgent:
    def __init__(
        self,
        model: str, 
        provider: str, 
        api_key: str,
        output_callback: Callable, 
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
    ):
        if model == "omniparser + gpt-4o":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1":
            self.model = "deepseek-r1-distill-llama-70b"
        elif model == "omniparser + qwen2.5vl":
            self.model = "qwen2.5-vl-72b-instruct"
        elif model == "omniparser + o1":
            self.model = "o1"
        elif model == "omniparser + o3-mini":
            self.model = "o3-mini"
        elif model == "omniparser + ollama":
            self.model = "deepseek-r1:8b"
        elif model == "omniparser + lmstudio":
            self.model = "qwen2.5-7b-instruct"
        else:
            raise ValueError(f"Model {model} not supported")
        

        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0

        self.system = ''
           
    def __call__(self, messages: list, parsed_screen: list[str, list, dict]):
        self.step_count += 1
        image_base64 = parsed_screen['original_screenshot_base64']
        latency_omniparser = parsed_screen['latency']
        self.output_callback(f'-- Step {self.step_count}: --', sender="bot")
        screen_info = str(parsed_screen['screen_info'])
        screenshot_uuid = parsed_screen['screenshot_uuid']
        screen_width, screen_height = parsed_screen['width'], parsed_screen['height']

        boxids_and_labels = parsed_screen["screen_info"]
        system = self._get_system_prompt(boxids_and_labels)

        # drop looping actions msg, byte image etc
        planner_messages = messages
        _remove_som_images(planner_messages)
        _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)

        if isinstance(planner_messages[-1], dict):
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_{screenshot_uuid}.png")
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_som_{screenshot_uuid}.png")

        start = time.time()
        if "gpt" in self.model or "o1" in self.model or "o3-mini" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                provider_base_url="https://api.openai.com/v1",
                temperature=0,
            )
            print(f"oai token usage: {token_usage}")
            self.total_token_usage += token_usage
            if 'gpt' in self.model:
                self.total_cost += (token_usage * 2.5 / 1000000)
            elif 'o1' in self.model:
                self.total_cost += (token_usage * 15 / 1000000)
            elif 'o3-mini' in self.model:
                self.total_cost += (token_usage * 1.1 / 1000000)
        elif "deepseek-r1:8b" in self.model:
            vlm_response, token_usage = self.run_ollama_interleaved(
                messages=planner_messages,
                system=system,
                model_name="deepseek-r1:8b",
                max_tokens=self.max_tokens,
            )
            print(f"ollama token usage: {token_usage}")
            self.total_token_usage += token_usage
            # Ollama is free so no cost tracking needed
        elif "qwen2.5-7b-instruct" in self.model:  # LM Studio
            vlm_response, token_usage = self.run_lmstudio_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                max_tokens=self.max_tokens,
            )
            print(f"lmstudio token usage: {token_usage}")
            self.total_token_usage += token_usage
            # LM Studio is free so no cost tracking needed
        elif "r1" in self.model:
            vlm_response, token_usage = run_groq_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
            )
            print(f"groq token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.99 / 1000000)
        elif "qwen" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=min(2048, self.max_tokens),
                provider_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0,
            )
            print(f"qwen token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 2.2 / 1000000)  # https://help.aliyun.com/zh/model-studio/getting-started/models?spm=a2c4g.11186623.0.0.74b04823CGnPv7#fe96cfb1a422a
        else:
            raise ValueError(f"Model {self.model} not supported")
        latency_vlm = time.time() - start
        self.output_callback(f"LLM: {latency_vlm:.2f}s, OmniParser: {latency_omniparser:.2f}s", sender="bot")

        print(f"{vlm_response}")
        
        if self.print_usage:
            print(f"Total token so far: {self.total_token_usage}. Total cost so far: $USD{self.total_cost:.5f}")
        
        vlm_response_json = extract_data(vlm_response, "json")
        vlm_response_json = json.loads(vlm_response_json)

        img_to_show_base64 = parsed_screen["som_image_base64"]
        if "Box ID" in vlm_response_json:
            try:
                bbox = parsed_screen["parsed_content_list"][int(vlm_response_json["Box ID"])]["bbox"]
                vlm_response_json["box_centroid_coordinate"] = [int((bbox[0] + bbox[2]) / 2 * screen_width), int((bbox[1] + bbox[3]) / 2 * screen_height)]
                img_to_show_data = base64.b64decode(img_to_show_base64)
                img_to_show = Image.open(BytesIO(img_to_show_data))

                draw = ImageDraw.Draw(img_to_show)
                x, y = vlm_response_json["box_centroid_coordinate"] 
                radius = 10
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
                draw.ellipse((x - radius*3, y - radius*3, x + radius*3, y + radius*3), fill=None, outline='red', width=2)

                buffered = BytesIO()
                img_to_show.save(buffered, format="PNG")
                img_to_show_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except:
                print(f"Error parsing: {vlm_response_json}")
                pass
        self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">', sender="bot")
        self.output_callback(
                    f'<details>'
                    f'  <summary>Parsed Screen elemetns by OmniParser</summary>'
                    f'  <pre>{screen_info}</pre>'
                    f'</details>',
                    sender="bot"
                )
        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'

        # construct the response so that anthropicExcutor can execute the tool
        response_content = [BetaTextBlock(text=vlm_plan_str, type='text')]
        if 'box_centroid_coordinate' in vlm_response_json:
            move_cursor_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': 'mouse_move', 'coordinate': vlm_response_json["box_centroid_coordinate"]},
                                            name='computer', type='tool_use')
            response_content.append(move_cursor_block)

        if vlm_response_json["Next Action"] == "None":
            print("Task paused/completed.")
        elif vlm_response_json["Next Action"] == "type":
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': vlm_response_json["Next Action"], 'text': vlm_response_json["value"]},
                                        name='computer', type='tool_use')
            response_content.append(sim_content_block)
        else:
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': vlm_response_json["Next Action"]},
                                            name='computer', type='tool_use')
            response_content.append(sim_content_block)
        response_message = BetaMessage(id=f'toolu_{uuid.uuid4()}', content=response_content, model='', role='assistant', type='message', stop_reason='tool_use', usage=BetaUsage(input_tokens=0, output_tokens=0))
        return response_message, vlm_response_json

    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)

    def _get_system_prompt(self, screen_info: str = ""):
        main_section = f"""
You are using a Windows device.
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
You can only interact with the desktop GUI (no terminal or application menu access).

You may be given some history plan and actions, this is the response from the previous loop.
You should carefully consider your plan base on the task, screenshot, and history actions.

Here is the list of all detected bounding boxes by IDs on the screen and their description:{screen_info}

Your available "Next Action" only include:
- type: types a string of text.
- double_click: move mouse to box id and double clicks.
- left_click: move mouse to box id and left clicks.
- right_click: move mouse to box id and right clicks.
- hover: move mouse to box id.
- scroll_up: scrolls the screen up to view previous content.
- scroll_down: scrolls the screen down, when the desired button is not visible, or you need to see more content. 
- wait: waits for 1 second for the device to load or respond.

Based on the visual information from the screenshot image and the detected bounding boxes, please determine the next action, the Box ID you should operate on (if action is one of 'type', 'hover', 'scroll_up', 'scroll_down', 'wait', there should be no Box ID field), and the value (if the action is 'type') in order to complete the task.

Output format:
```json
{{
    "Reasoning": str, # describe what is in the current screen, taking into account the history, then describe your step-by-step thoughts on how to achieve the task, choose one action from available actions at a time.
    "Next Action": "action_type, action description" | "None" # one action at a time, describe it in short and precisely. 
    "Box ID": n,
    "value": "xxx" # only provide value field if the action is type, else don't include value key
}}
```

One Example:
```json
{{  
    "Reasoning": "The current screen shows google result of amazon, in previous action I have searched amazon on google. Then I need to click on the first search results to go to amazon.com.",
    "Next Action": "left_click",
    "Box ID": m
}}
```

Another Example:
```json
{{
    "Reasoning": "The current screen shows the front page of amazon. There is no previous action. Therefore I need to type "Apple watch" in the search bar.",
    "Next Action": "type",
    "Box ID": n,
    "value": "Apple watch"
}}
```

Another Example:
```json
{{
    "Reasoning": "The current screen does not show 'submit' button, I need to scroll down to see if the button is available.",
    "Next Action": "scroll_down",
}}
```

IMPORTANT NOTES:
1. You should only give a single action at a time.

"""
        thinking_model = "deepseek-r1:8b" or "qwen2.5-7b-instruct" or "r1" in self.model
        if not thinking_model:
            main_section += """
2. You should give an analysis to the current screen, and reflect on what has been done by looking at the history, then describe your step-by-step thoughts on how to achieve the task.

"""
        else:
            main_section += """
2. In <think> XML tags give an analysis to the current screen, and reflect on what has been done by looking at the history, then describe your step-by-step thoughts on how to achieve the task. In <output> XML tags put the next action prediction JSON.

"""
        main_section += """
3. Attach the next action prediction in the "Next Action".
4. You should not include other actions, such as keyboard shortcuts.
5. When the task is completed, don't complete additional actions. You should say "Next Action": "None" in the json field.
6. The tasks involve buying multiple products or navigating through multiple pages. You should break it into subgoals and complete each subgoal one by one in the order of the instructions.
7. avoid choosing the same action/elements multiple times in a row, if it happens, reflect to yourself, what may have gone wrong, and predict a different action.
8. If you find text like username and password then you should say "Next Action": "None" in the json field.
9. If you are prompted with login information page or captcha page, or you think it need user's permission to do the next action, you should say "Next Action": "None" in the json field.
""" 

        return main_section

    def run_ollama_interleaved(
        self,
        messages: list,
        system: str,
        model_name: str,
        max_tokens: int = 4096,
    ) -> Tuple[str, int]:
        """
        Run a chat completion through Ollama's API, ignoring any images in the messages.
        """
        print("Starting Ollama interleaved...")
        
        # Start with system message as user message
        final_messages = [{"role": "user", "content": system}]

        # Process all messages in one pass
        if isinstance(messages, list):
            for item in messages:
                if not isinstance(item, dict):
                    final_messages.append({"role": "user", "content": str(item)})
                    continue
                
                # Extract all non-image text in one list comprehension
                text_contents = [
                    str(cnt) for cnt in item.get("content", [])
                    if isinstance(cnt, str) and not is_image_path(cnt)
                    or not isinstance(cnt, str)
                ]
                
                if text_contents:
                    final_messages.append({
                        "role": "user",
                        "content": " ".join(text_contents)
                    })
        
        elif isinstance(messages, str):
            final_messages.append({"role": "user", "content": messages})

        # Prepare the request with optimized settings
        payload = {
            "model": model_name,
            "messages": final_messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.8,
            }
        }

        print(f"Sending request with {len(final_messages)} messages...")

        try:
            # Make request to local Ollama server with increased timeout
            response = requests.post(
                'http://localhost:11434/api/chat',
                json=payload,
                timeout=60  # Increase timeout to 60 seconds
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            response_text = result.get('message', {}).get('content', '')
            
            # Extract JSON more efficiently
            if "```json" in response_text:
                try:
                    json_text = extract_data(response_text, "json")
                    response_text = json_text
                except:
                    pass
            
            # Format non-JSON responses more efficiently
            if not response_text.strip().startswith('{'):
                response_text = f'{{"Reasoning": "{response_text.strip().replace('"', '\\"')}", "Next Action": "None"}}'
            
            return response_text, len(response_text) // 4

        except requests.Timeout:
            error_msg = '{"Reasoning": "Ollama request timed out. Please try again.", "Next Action": "None"}'
            print("Ollama request timed out")
            return error_msg, 0
        except requests.ConnectionError:
            error_msg = '{"Reasoning": "Could not connect to Ollama server. Make sure it is running.", "Next Action": "None"}'
            print("Could not connect to Ollama server")
            return error_msg, 0
        except Exception as e:
            error_msg = f'{{"Reasoning": "Error calling Ollama API: {str(e)}", "Next Action": "None"}}'
            print(f"Error calling Ollama API: {e}")
            return error_msg, 0
        
    def run_lmstudio_interleaved(
            self,
            messages: list,
            system: str,
            model_name: str,
            max_tokens: int = 4096,
        ) -> Tuple[str, int]:
            """
            Run a chat completion through LM Studio's API for reasoning model (no image handling).
            Following same pattern as Groq implementation.
            """
            print("Starting LM Studio interleaved...")
            
            # Start with system message as user message like Groq
            final_messages = [{"role": "user", "content": system}]

            # Process messages similar to Groq implementation
            if isinstance(messages, list):
                for item in messages:
                    if isinstance(item, dict):
                        # For dict items, concatenate all text content, ignoring images
                        text_contents = []
                        for cnt in item.get("content", []):
                            if isinstance(cnt, str):
                                if not is_image_path(cnt):  # Skip image paths
                                    text_contents.append(cnt)
                            else:
                                text_contents.append(str(cnt))
                        
                        if text_contents:  # Only add if there's text content
                            message = {"role": "user", "content": " ".join(text_contents)}
                            final_messages.append(message)
                    else:  # str
                        message = {"role": "user", "content": str(item)}
                        final_messages.append(message)
            
            elif isinstance(messages, str):
                final_messages.append({"role": "user", "content": messages})

            # Prepare the request exactly as shown in curl command
            payload = {
                "model": model_name,
                "messages": final_messages,
                "temperature": 0.8,  # Match Groq's temperature
                "max_tokens": -1,  # Use -1 for no limit as shown in curl
                "stream": False
            }

            try:
                # Make request to LM Studio server
                response = requests.post(
                    'http://localhost:1234/v1/chat/completions',
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60
                )
                
                # Check if response is empty before trying to parse it
                if not response.content:
                    print("Empty response received from LM Studio")
                    return '{"Reasoning": "Empty response from LM Studio server", "Next Action": "None"}', 0
                    
                # Check status code
                if response.status_code != 200:
                    print(f"LM Studio returned error status: {response.status_code}")
                    return f'{{"Reasoning": "LM Studio error: HTTP {response.status_code}", "Next Action": "None"}}', 0
                    
                # Try to parse JSON response, handling empty or malformed responses
                try:
                    result = response.json()
                except requests.exceptions.JSONDecodeError as e:
                    print(f"Invalid JSON response from LM Studio: {e}")
                    print(f"Raw response content: {response.content}")  # Log the raw response
                    return f'{{"Reasoning": "Invalid JSON response from LM Studio: {str(e)}", "Next Action": "None"}}', 0
                    
                if not result or 'choices' not in result:
                    print(f"Invalid response structure from LM Studio: {result}")
                    return '{"Reasoning": "Invalid response structure from LM Studio", "Next Action": "None"}', 0
                
                response_text = result['choices'][0]['message']['content']
                print(f"Raw response: {response_text}")  # Print full response for debugging

                # Process response similar to Groq - handle think/output tags
                final_answer = response_text.split('</think>\n')[-1] if '</think>' in response_text else response_text
                final_answer = final_answer.replace("<output>", "").replace("</output>", "")

                # Clean up the response text to avoid invalid characters
                final_answer = final_answer.replace('\n', ' ').replace('\r', '')  # Remove newlines and carriage returns

                # Try to extract and fix JSON
                try:
                    if "```json" in final_answer:
                        json_text = extract_data(final_answer, "json")
                        try:
                            # Validate the JSON structure
                            parsed_json = json.loads(json_text)
                            # Ensure all required fields are present and handle truncation
                            if "Reasoning" not in parsed_json:
                                parsed_json["Reasoning"] = "No reasoning provided"
                            if "Next Action" not in parsed_json:
                                parsed_json["Next Action"] = "wait"
                            # Handle truncated actions
                            if parsed_json["Next Action"].startswith("double"):
                                parsed_json["Next Action"] = "double_click"
                            # Ensure Box ID is present when needed
                            if parsed_json["Next Action"] in ["left_click", "right_click", "double_click", "hover"]:
                                if "Box ID" not in parsed_json:
                                    parsed_json["Box ID"] = None
                                    parsed_json["Next Action"] = "wait"
                                    parsed_json["Reasoning"] += " (Error: Missing Box ID for click action)"
                            # Clean up value field
                            if "value" in parsed_json and not parsed_json["value"]:
                                del parsed_json["value"]
                            final_answer = json.dumps(parsed_json)
                            print("Valid JSON extracted and normalized from response")
                        except json.JSONDecodeError as e:
                            print(f"Invalid JSON in response, falling back to text: {e}")
                            final_answer = f'{{"Reasoning": "{json_text.strip().replace('"', '\\"')}", "Next Action": "None"}}'
                    else:
                        final_answer = f'{{"Reasoning": "{final_answer.strip().replace('"', '\\"')}", "Next Action": "None"}}'
                        print("Wrapped plain text in JSON format")

                    # Final validation of JSON
                    try:
                        json.loads(final_answer)
                    except json.JSONDecodeError as e:
                        print(f"Final JSON validation failed: {e}")
                        # Ensure we always return valid JSON
                        final_answer = '{"Reasoning": "Error processing response into valid JSON", "Next Action": "None"}'
                    
                except Exception as e:
                    print(f"Error processing response text: {e}")
                    final_answer = '{"Reasoning": "Error processing response", "Next Action": "None"}'
                
                # Get token usage from response
                token_usage = result.get('usage', {}).get('total_tokens', len(final_answer) // 4)
                
                return final_answer, token_usage

            except requests.Timeout:
                error_msg = '{"Reasoning": "LM Studio request timed out. Please try again.", "Next Action": "None"}'
                print("LM Studio request timed out")
                return error_msg, 0
            except requests.ConnectionError:
                error_msg = '{"Reasoning": "Could not connect to LM Studio server. Make sure it is running.", "Next Action": "None"}'
                print("Could not connect to LM Studio server")
                return error_msg, 0
            except Exception as e:
                error_msg = f'{{"Reasoning": "Error calling LM Studio API: {str(e)}", "Next Action": "None"}}'
                print(f"Error calling LM Studio API: {e}")
                if 'response' in locals():
                    print(f"Response content: {response.content}")
                else:
                    print("No response object available")
                return error_msg, 0

def _remove_som_images(messages):
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            msg["content"] = [
                cnt for cnt in msg_content 
                if not (isinstance(cnt, str) and 'som' in cnt and is_image_path(cnt))
            ]


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place
    """
    if images_to_keep is None:
        return messages

    total_images = 0
    for msg in messages:
        for cnt in msg.get("content", []):
            if isinstance(cnt, str) and is_image_path(cnt):
                total_images += 1
            elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                for content in cnt.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        total_images += 1

    images_to_remove = total_images - images_to_keep
    
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            new_content = []
            for cnt in msg_content:
                # Remove images from SOM or screenshot as needed
                if isinstance(cnt, str) and is_image_path(cnt):
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                # VLM shouldn't use anthropic screenshot tool so shouldn't have these but in case it does, remove as needed
                elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                    new_tool_result_content = []
                    for tool_result_entry in cnt.get("content", []):
                        if isinstance(tool_result_entry, dict) and tool_result_entry.get("type") == "image":
                            if images_to_remove > 0:
                                images_to_remove -= 1
                                continue
                        new_tool_result_content.append(tool_result_entry)
                    cnt["content"] = new_tool_result_content
                # Append fixed content to current message's content list
                new_content.append(cnt)
            msg["content"] = new_content