from utils.prompt_interp import system_prompt_interp

test = system_prompt_interp(file_name="feedback_prompt.txt", strict=True, grounding_content="BLA", last_action_str="CLIC", previous_action="TEST", task_process_str="TRI")
print(test)