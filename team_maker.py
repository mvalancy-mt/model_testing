"""
team_maker.py

This script automatically generates an infinite series of team organizations with backstories, names,
descriptions, and detailed prompts for each role using locally hosted language models (LLMs) configured with LM Studio.
The purpose is to dynamically create imaginative project teams with unique roles, backstories, and responsibilities 
that are tailored to a generated problem or project.

How it works:
1. **Local LLM Integration**:
   - The script interacts with locally hosted LLMs (e.g., Meta-Llama models) set up via LM Studio.
   - Multiple LLM endpoints are supported, allowing load distribution across instances.
   - Queries are sent to these LLMs via an HTTP API to generate role names, descriptions, and detailed prompts.

2. **Problem Generation**:
   - Before generating a team, the LLM is tasked with generating a short problem description. 
   - This problem can be a social issue, software or hardware problem, or even a patent-related problem. 
   - The problem description begins with either 'Problem:' or 'Project:' and serves as the basis for creating the team.

3. **Team Generation**:
   - Based on the generated problem, the script creates teams of 2-6 unique roles.
   - For each role, the LLM generates:
     a. A short backstory with a unique name, skills, and description of how the individual fits the team.
     b. A detailed prompt that describes the specific responsibilities, collaboration with others, and example tasks.

4. **Retries and Prompt Optimization**:
   - If a role's details are incomplete, the script retries until full descriptions and prompts are generated.
   - This allows optimization of prompt quality and improves the success rate of generating team members' details.

5. **Saving and Logging**:
   - Generated roles and prompts are saved in JSON format, with individual files for each role and a team summary.
   - All actions and issues (such as retries or errors) are logged for tracking purposes.

6. **Continuous Execution with Graceful Shutdown**:
   - The script is designed to run continuously, generating problems and teams until stopped manually (Ctrl+C).
   - It handles shutdown gracefully, ensuring logs are closed and proper termination is handled without abrupt interruption.

Usage Example:
1. Set up locally hosted LLM instances with LM Studio.
2. Run the script, which will generate a problem, create a team for the problem, and display the results.
3. Generated teams, along with their descriptions and prompts, are saved to files for later review or use.

Dependencies:
- Python 3.x
- aiohttp for asynchronous HTTP requests
- LM Studio with locally hosted LLM instances
"""

import sys      # Provides access to system-specific parameters and functions
import signal   # Allows handling of system signals, like Ctrl+C (SIGINT)
import asyncio  # Enables asynchronous I/O operations in Python
import aiohttp  # Library for making asynchronous HTTP requests
import json     # Provides methods for working with JSON data
import logging  # Handles logging of messages for debugging and information
import os       # Provides functions for interacting with the operating system
import random   # Implements random number generation functionality
from datetime import datetime  # For working with date and time
import re       # Regular expressions for pattern matching and text processing
import pdb      # For interactive debugging of Python code


# Setup Logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)  # Set to DEBUG level to track more granular details
sys.stdout.flush = lambda: None
fmt = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')

# Console handler for immediate print output
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(fmt)
log.addHandler(ch)

# File handler for logging to file
fh = logging.FileHandler('team_maker.log')
fh.setLevel(logging.DEBUG)

fh.setFormatter(fmt)
log.addHandler(fh)

class TeamMaker:

    def __init__(self, llm_endpoints):
        """
        Initialize the TeamMaker class.

        :param llm_endpoints: List of LLM endpoints to interact with.
        """
        self.roles = {}
        self.mem = {}
        self.llm_eps = llm_endpoints
        self.endpoint_assignments = {}  # To keep track of which role is assigned to which endpoint
        self.problem = "Create a self-replicating and self-improving machine that is represented by a friendly and approachable commitee"


    def assign_endpoint(self, role):
        """
        Assign an LLM endpoint to a specific role using round-robin distribution.

        :param role: The role to assign an endpoint to.
        :return: The assigned LLM endpoint.
        """
        # Cycle through the endpoints by using modulo to assign each role to a specific endpoint
        if role not in self.endpoint_assignments:
            idx = len(self.endpoint_assignments) % len(self.llm_eps)
            self.endpoint_assignments[role] = self.llm_eps[idx]
        return self.endpoint_assignments[role]


    async def get_res(self, role, msg, ep):
        """
        Get response from the assigned LLM endpoint for a given role.

        :param role: The role for which to get a response.
        :param msg: The message prompt sent to the LLM.
        :param ep: The LLM endpoint to query.
        :return: The response from the LLM or an error message in case of failure.
        """
        max_retries = 3  # Limit the number of retries to avoid infinite loops
        attempt = 0

        while attempt < max_retries:
            try:
                log.debug(f"Sending request to {ep['url']} for role {role} (attempt {attempt + 1})...")
                async with aiohttp.ClientSession() as sess:
                    hdrs = {'Content-Type': 'application/json', 'Authorization': f'Bearer {ep["key"]}'}
                    pl = {'model': ep["model"], 'messages': msg, 'temperature': 0.7}

                    async with sess.post(f'{ep["url"]}/chat/completions', headers=hdrs, json=pl) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            txt = data['choices'][0]['message']['content']
                            self.mem.setdefault(role, []).append(txt)
                            log.info(f"Response received for {role} from {ep['url']}")
                            return txt
                        else:
                            err = await resp.text()
                            log.error(f"Error {resp.status}: {err} from {ep['url']}")
                            raise aiohttp.ClientError(f"Error {resp.status}: {err}")

            except (aiohttp.ClientError, aiohttp.ClientConnectionError, asyncio.TimeoutError) as ex:
                log.error(f"Connection error with {ep['url']} for role {role}: {ex}")
                attempt += 1
                if attempt < max_retries:
                    log.info(f"Retrying {ep['url']} for role {role}... (attempt {attempt + 1})")
                    await asyncio.sleep(2)  # Wait for 2 seconds before retrying
                else:
                    log.error(f"Failed to reach {ep['url']} after {max_retries} attempts. Moving on.")
                    return f"Error in {role} with {ep['url']} after {max_retries} retries."



    async def generate_problem(self):
        """
        Generate a short description of a problem that needs to be solved.
        The problem can be a social issue, software problem, hardware problem, or patent-related problem.

        :return: The generated problem description.
        """
        
        while True:  # Loop until a valid problem is generated
            try:
                sys_msg_problem = {
                    "role": "system",
                    "content": (
                        "Generate a short description of a problem that needs to be solved. "
                        "The description should begin with either 'Problem:' or 'Project:' and be under 250 characters. "
                        "It should be related to a social issue, software problem, hardware problem, or patent-related problem. "
                        "The response must be concise, creative, and provide a real-world context for the problem."
                    )
                }
                user_msg_problem = {"role": "user", "content": "Please generate a problem description."}

                # Query all endpoints for the problem
                tasks = [self.get_res('problem', [sys_msg_problem, user_msg_problem], ep) for ep in self.llm_eps]
                results = await asyncio.gather(*tasks)

                # Check for the first valid problem generated
                for result in results:
                    if result and await self.is_valid_problem(result):  # Validate the problem
                        self.problem = result
                        log.info(f"Problem generated successfully: {result}")
                        return result  # Exit the loop and return the valid problem
            except Exception as e:
                log.error(f"Error generating problem: {e}")

            # If no valid problem was generated, log a warning and retry
            log.warning("No valid problem generated. Retrying...")



    async def is_valid_problem(self, problem):
        """
        Validate if the generated problem meets the required format.

        :param problem: The problem string to validate.
        :return: True if the problem is valid, otherwise False.
        """
        problem = problem.strip()
        
        # Check if the problem starts with "Problem:" or "Project:" and is under 250 characters
        if (problem.startswith("Problem:") or problem.startswith("Project:")) and len(problem) <= 250:
            return True
        return False


    async def gen_committee(self, initial_prompt=None):
        """
        Generate a committee of roles based on the initial project prompt.

        :param initial_prompt: The project prompt to generate the committee for.
        :return: The generated committee with roles and descriptions.
        """
        if initial_prompt:
            self.problem = initial_prompt
        committee_size = random.randint(2, 6)
        log.info(f"Generating committee of size {committee_size} for project: {self.problem}")
        roles = await self.generate_roles(committee_size)
        while not roles:
            log.warning("No roles returned, retrying role generation...")
            roles = await self.generate_roles(committee_size)

        log.info(f"Roles generated: {list(roles.keys())}")
        for role in self.roles.keys():
            description, detailed_prompt = None, None
            while description is None or detailed_prompt is None:
                log.debug(f"Generating details for role: {role}")
                description, detailed_prompt = await self.generate_role_details(role, self.problem)
            self.roles[role]["description"] = description
            self.roles[role]["detailed_prompt"] = detailed_prompt
            log.info(f"Details for {role} generated successfully")

        return self.roles


    async def generate_roles(self, committee_size):
        """
        Generate a list of roles for the committee.

        :param committee_size: The number of roles to generate.
        :return: A dictionary of roles.
        """

        log.debug(f"Attempting to generate {committee_size} roles for the committee...")
        while True:
            try:
                sys_msg = {
                    "role": "system",
                    "content": (
                        f"Generate a committee with {committee_size} distinct roles for this project. "
                        "Respond in this format: Role: [Role Name], but only list the short names of the roles "
                        "with no other details or conversation. Reply strictly with the role names. Do not include punctuation in the response. "
                        "Include one role per line and do not repeat roles"
                    )
                }
                user_msg = {"role": "user", "content": f"Objective Description: {self.problem}"}

                # Send the role generation request to a random endpoint
                assigned_ep = random.choice(self.llm_eps)
                roles_response = await self.get_res('roles', [sys_msg, user_msg], assigned_ep)

                if roles_response:
                    new_roles = self.parse_roles_list(roles_response)
                    log.debug(f"Roles parsed: {new_roles}")
                    for role in new_roles:
                        if role not in self.roles:
                            self.roles[role] = {"description": None, "detailed_prompt": None}

                        if len(self.roles) >= committee_size:
                            log.info(f"Successfully generated {committee_size} roles.")
                            return self.roles
            except Exception as e:
                log.error(f"Error generating roles: {e}")
                continue


    async def generate_role_details(self, role, project_info):
        """
        Generate detailed role descriptions and prompts, retrying until successful.

        :param role: The role to generate the description and prompt for.
        :param project_info: Information about the project to provide context.
        :return: The role's description and detailed prompt.
        """
        attempt = 0

        # Retry until both description and detailed prompt are generated
        while True:
            attempt += 1
            log.info(f"Attempt {attempt} to generate details for role: {role}")

            try:
                tasks = []  # Create an empty task list
                description, detailed_prompt = None, None  # Reset description and prompt

                # If description is missing, create a task to generate it
                if self.roles[role].get("description") is None:
                    sys_msg_desc = {
                        "role": "system",
                        "content": (
                            f"Generate a short description for the role '{role}', "
                            "focusing on their personal backstory, abilities, and how they collaborate with the team. "
                            f"In the context of the following project:\n\nProject: {project_info}\n\n"
                            "Do not use typical job listing language. "
                            "Instead, create an imaginative backstory for this team member, "
                            "including their expertise, how they joined the team, and their unique skills. "
                            "Provide the response in the following format:\n\n"
                            "Name: [Imaginative Name]\n"
                            "Backstory: [Short imaginative backstory about their journey to the team]\n"
                            "Appearance: [Short imaginative but detailed description of the character's appearance]\n"
                            "Abilities: [Describe their unique abilities and how they collaborate]\n"
                            "Do not include project-specific responsibilities or tasks."
                        )
                    }
                    user_msg_desc = {"role": "user", "content": f"Role: {role}"}
                    tasks.append(self.get_res(f'{role}_desc', [sys_msg_desc, user_msg_desc], random.choice(self.llm_eps)))

                # If detailed prompt is missing, create a task to generate it
                if self.roles[role].get("detailed_prompt") is None:
                    sys_msg_prompt = {
                        "role": "system",
                        "content": (
                            f"Generate a detailed prompt for the role '{role}' "
                            "to assist with the following project:\n\n"
                            f"Project: {project_info}\n\n"
                            "This should describe their specific responsibilities, "
                            "how they will collaborate with other team members, and include examples of how "
                            "their abilities will be used in key project tasks. "
                            "Provide the response in the following format:\n\n"
                            "Project Role: [Role Name]\n"
                            "Responsibilities: [Detailed explanation of the tasks this role will perform]\n"
                            "Collaboration: [How this role will collaborate with other team members]\n"
                            "Example Task: [An example scenario showing how their skills are applied in the project]\n"
                            "Do not include any backstory or personality traits."
                        )
                    }
                    user_msg_prompt = {"role": "user", "content": f"Project Role: {role}"}
                    tasks.append(self.get_res(f'{role}_prompt', [sys_msg_prompt, user_msg_prompt], random.choice(self.llm_eps)))

                # Execute all tasks in parallel
                if tasks:
                    responses = await asyncio.gather(*tasks)
                    log.debug(f"Responses received for role {role}: {responses}")

                    # Parse each response to extract either description or detailed prompt
                    for response in responses:
                        if response:
                            if 'Name:' in response and self.roles[role].get("description") is None:
                                description = self.parse_short_description(response)
                                if description:
                                    self.roles[role]["description"] = description
                                    log.info(f"Description for {role} generated successfully.")

                            if 'Project Role:' in response and self.roles[role].get("detailed_prompt") is None:
                                detailed_prompt = self.parse_detailed_prompt(response)
                                if detailed_prompt:
                                    self.roles[role]["detailed_prompt"] = detailed_prompt
                                    log.info(f"Detailed prompt for {role} generated successfully.")

            except Exception as e:
                log.error(f"Error generating details for {role} on attempt {attempt}: {e}")
                continue  # Retry on failure

            # Return if both description and detailed prompt are available
            if self.roles[role].get("description") and self.roles[role].get("detailed_prompt"):
                log.info(f"Successfully generated details for role: {role} on attempt {attempt}")
                return self.roles[role]["description"], self.roles[role]["detailed_prompt"]


    def parse_roles_list(self, response):
        """
        Parse a list of roles from the LLM response.

        :param response: The LLM response containing role information.
        :return: A list of roles parsed from the response.
        """
        pattern = r"(?:Role\s*:|^\d+\.\s*)(.*)"
        matches = re.findall(pattern, response)
        if matches:
            return [role.replace(":", ",").strip() for role in matches if role.strip()]

        roles = [line.strip() for line in response.splitlines() if line.strip()]
        return roles


    def parse_short_description(self, response):
        """
        Parse a short description of a role from the LLM response.

        :param response: The LLM response containing the role description.
        :return: A dictionary containing the parsed description elements (Name, Backstory, Appearance, Abilities, etc.).
        """
        description = {}
        # Updated regex patterns to handle the new prompt structure
        name_pattern = r"Name:\s*(.*?)\s*(?=Backstory:|$)"
        backstory_pattern = r"Backstory:\s*(.*?)\s*(?=Appearance:|$)"
        appearance_pattern = r"Appearance:\s*(.*?)\s*(?=Abilities:|$)"
        abilities_pattern = r"Abilities:\s*(.*?)\s*(?=Role Contribution:|$|$)"  # Handles missing Role Contribution section
        contribution_pattern = r"Role Contribution:\s*(.*)"  # Role Contribution is optional, so it may not be present

        # Extract Name
        def clean_field(field):
            # Clean leading/trailing '**' and '\n\n' or '\n'
            return field.strip('**').strip('\n').strip()

        # Extract Name
        name_match = re.search(name_pattern, response, re.DOTALL)
        if name_match:
            description['Name'] = clean_field(name_match.group(1))
        else:
            return None

        # Extract Backstory
        backstory_match = re.search(backstory_pattern, response, re.DOTALL)
        if backstory_match:
            description['Backstory'] = clean_field(backstory_match.group(1))
        else:
            return None

        # Extract Appearance
        appearance_match = re.search(appearance_pattern, response, re.DOTALL)
        if appearance_match:
            description['Appearance'] = clean_field(appearance_match.group(1))
        else:
            return None

        # Extract Abilities
        abilities_match = re.search(abilities_pattern, response, re.DOTALL)
        if abilities_match:
            description['Abilities'] = clean_field(abilities_match.group(1))
        else:
            return None

        # Extract Role Contribution if it exists (optional)
        contribution_match = re.search(contribution_pattern, response, re.DOTALL)
        if contribution_match:
            description['Role Contribution'] = clean_field(contribution_match.group(1))

        return description


    def parse_detailed_prompt(self, response):
        """
        Parse a detailed prompt for a role from the LLM response.

        :param response: The LLM response containing the role prompt.
        :return: A dictionary containing the parsed detailed prompt (Responsibilities, Collaboration, Example Task, etc.).
        """
        prompt = {}
        role_pattern = r"Project Role:\s*(.*?)\s*(?=Responsibilities:|$)"
        responsibilities_pattern = r"Responsibilities:\s*(.*?)\s*(?=Collaboration:|$)"
        collaboration_pattern = r"Collaboration:\s*(.*?)\s*(?=Example Task:|$)"
        example_task_pattern = r"Example Task:\s*(.*)"

        def clean_field(field):
            # Clean leading/trailing '**' and '\n\n' or '\n'
            return field.strip('**').strip('\n').strip()

        # Extract Role
        role_match = re.search(role_pattern, response, re.DOTALL)
        if role_match:
            prompt['Project Role'] = clean_field(role_match.group(1))
        else:
            return None

        # Extract Responsibilities
        responsibilities_match = re.search(responsibilities_pattern, response, re.DOTALL)
        if responsibilities_match:
            prompt['Responsibilities'] = clean_field(responsibilities_match.group(1))
        else:
            return None

        # Extract Collaboration
        collaboration_match = re.search(collaboration_pattern, response, re.DOTALL)
        if collaboration_match:
            prompt['Collaboration'] = clean_field(collaboration_match.group(1))
        else:
            return None

        # Extract Example Task
        example_task_match = re.search(example_task_pattern, response, re.DOTALL)
        if example_task_match:
            prompt['Example Task'] = clean_field(example_task_match.group(1))
        else:
            return None

        return prompt


    async def save_prompts_to_file(self):
        """
        Save all generated role prompts to JSON files.
        """
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        p_dir = f"prompts/{t}"
        os.makedirs(p_dir, exist_ok=True)

        for role, data in self.roles.items():
            filename = f'{p_dir}/{role}_prompt_{t}.json'
            role_data = {
                "name": role,
                "description": data.get("description", {}),
                "prompt": data.get("detailed_prompt", "")
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(role_data, f, ensure_ascii=False, indent=4)

        log.info("Prompts saved successfully.")


    def show_team(self):
        """
        Display a summary of the generated team and save the details to a JSON file, including the problem.
        """
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        p_dir = f"output/"
        os.makedirs(p_dir, exist_ok=True)

        # Prepare the final data to include the problem and the team roles
        team_data = {
            "problem": self.problem,
            "roles": self.roles
        }

        # Save the team data to a JSON file
        team_file_path = f'{p_dir}/team_summary_{t}.json'
        with open(team_file_path, 'w') as f:
            json.dump(team_data, f, indent=4)
        
        log.info(f"Team details (including problem) saved to {team_file_path}")

        # Display the team summary
        print("\n--- Team Summary ---")
        print(f"Problem: {self.problem}")
        for role, details in self.roles.items():
            description = details.get('description', {})
            prompt = details.get('detailed_prompt', {})
            print(f"\nRole: {role}")
            print(f"Description: {description.get('Name', 'No Name Provided')}")
            print(f"Backstory: {description.get('Backstory', 'No Backstory Provided')}")
            print(f"Abilities: {description.get('Abilities', 'No Abilities Provided')}")
            print(f"Role Contribution: {description.get('Role Contribution', 'No Role Contribution Provided')}")
            print(f"Responsibilities: {prompt.get('Responsibilities', 'No Responsibilities Provided')}")
            print(f"Collaboration: {prompt.get('Collaboration', 'No Collaboration Info Provided')}")
            print(f"Example Task: {prompt.get('Example Task', 'No Example Task Provided')}")
        print("\n-----------------------")


def handle_shutdown(signal_received, frame):
    """
    Gracefully handle shutdown on Ctrl+C.
    """
    log.info("Shutdown requested...exiting program.")
    sys.exit(0)


def show_welcome_message():
    """
    Display an ASCII art welcome message for the Team Maker program.
    """
    print(r"""
   ******************************************************
   *                                                    *
   *              Welcome to Team Maker!                *
   *                                                    *
   *                by Matthew Valancy                  *
   *                                                    *
   ******************************************************

 
   ▄▄▄▄▀ ▄███▄   ██   █▀▄▀█     █▀▄▀█ ██   █  █▀ ▄███▄   █▄▄▄▄ 
▀▀▀ █    █▀   ▀  █ █  █ █ █     █ █ █ █ █  █▄█   █▀   ▀  █  ▄▀ 
    █    ██▄▄    █▄▄█ █ ▄ █     █ ▄ █ █▄▄█ █▀▄   ██▄▄    █▀▀▌  
   █     █▄   ▄▀ █  █ █   █     █   █ █  █ █  █  █▄   ▄▀ █  █  
  ▀      ▀███▀      █    █         █     █   █   ▀███▀     █   
                   █    ▀         ▀     █   ▀             ▀    
                  ▀                    ▀                       


      An automatic team and role generation program.
   """)

if __name__ == "__main__":
    llm_endpoints = [
        {"url": "http://localhost:1234/v1", "key": "lm_studio", "model": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"},
        {"url": "http://192.168.1.4:1234/v1", "key": "lm_studio", "model": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"}
    ]

    signal.signal(signal.SIGINT, handle_shutdown)

    show_welcome_message()

    while True:
        try:
            # Initialize TeamMaker with LLM endpoints
            team_maker = TeamMaker(llm_endpoints)

            # Generate a problem for the team to solve
            problem = asyncio.run(team_maker.generate_problem())
            log.info(f"Problem generated: {problem}")

            # Generate the committee based on the problem
            asyncio.run(team_maker.gen_committee(problem))

            # Show the generated team
            team_maker.show_team()

        except Exception as e:
            # Log the error and continue running
            log.error(f"Error encountered: {e}")
            log.info("Retrying in 5 seconds...")
            asyncio.sleep(5)  # Wait for 5 seconds before retrying