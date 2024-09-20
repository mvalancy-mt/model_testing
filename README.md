# model_testing

Welcome to the `model_testing` repository! This repository is dedicated to general scripts for testing and experimenting with various AI/ML models, including tools for generating dynamic teams, solving problems, and more.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
  - [Windows](#windows)
  - [Linux/macOS](#linuxmacos)
- [License](#license)

## Overview

`model_testing` contains Python scripts that interact with local language model (LLM) servers set up with [LM Studio](https://lmstudio.ai) to generate dynamic teams, projects, and solutions. The initial project included here is a **team committee generator** that uses LLMs to create team roles, backstories, and problem-solving committees.

The current project is the `team_maker.py` script, which uses the LLM to generate a team of fictional roles, each with detailed descriptions and collaboration strategies, aimed at solving a generated problem related to social issues, software, hardware, or patents.

## Features

- **Local LLM integration**: Uses local language models served through LM Studio.
- **Team generation**: Automatically creates teams with imaginative roles and prompts for specific project needs.
- **Customizable problems**: LLM generates unique problems for the team to solve, such as social issues or technical problems.
- **Dynamic retries**: The system keeps retrying until valid responses for team roles and problems are generated.
- **Logging**: Detailed logging is available, including team summaries and problem information.

## Setup

### Requirements

Before using the project, make sure you have the following installed:

- Python 3.x
- `aiohttp` (Python asynchronous HTTP client)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/model_testing.git
   cd model_testing
