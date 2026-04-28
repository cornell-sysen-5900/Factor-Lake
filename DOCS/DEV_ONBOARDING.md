# Factor Lake Portfolio: New Member Onboarding Guide

A concise guide to getting access, setting up your environment, and understanding how work moves through the project.

## ⚠️ Important

You cannot begin contributing until the professor adds you to all required workspaces: GitHub, Supabase, Trello, and Slack.

## 1. Purpose of this guide

This document is meant to help new members get productive quickly without needing to search across multiple tools for the basics. It covers what platforms the project uses, how to set up your access, where to find documentation, how the Trello board works, and how the team's weekly workflow moves from planning to implementation, review, demo, and completion.

## 2. The tools used in the project

| Tool | Primary purpose | What a new member should know |
|------|-----------------|------------------------------|
| GitHub Repo (Factor-Lake) | Source code and documentation | Use it to clone the repository, create branches, submit pull requests, and read project docs such as CONTRIBUTING.md and DEPLOYMENT.md and SUPABASE_SETUP.md |
| Supabase | Backend database | Most members will not work in Supabase frequently, but it is used when new data must be added or new tables need to be created and populated. |
| Trello | Sprint planning and workflow tracking | The Trello board shows where work lives, what is planned next, who is actively working on something, and what is waiting for review or demo. |
| Slack | Team communication | Slack is where new members ask questions, coordinate with current members, and stay aligned during the semester |

## 3. Before you start

Your first step is to wait until the professor has added you to all four project workspaces. Once access is granted, verify that you can sign in to GitHub, Supabase, Trello, and Slack. If one of these is missing, receive that first before trying to contribute, because the project workflow depends on all of them.

### First-day checklist

- Confirm access to the Factor-Lake GitHub repository.
- Confirm access to the Supabase project.
- Confirm access to the Trello board.
- Confirm access to the Slack workspace.
- Review the semester Slack channel and the Trello Reference list before starting work.

## 4. Slack setup and communication norms

After joining Slack, someone on the team should create the semester channel using the format fall-20XX or spring-20XX. This channel becomes the main place for that semester's communication. New members should use it to introduce themselves, ask questions, clarify requirements, and coordinate with existing members whenever something in the codebase or workflow is unclear.

Do not wait too long to ask questions. Slack is intended to reduce confusion early and make it easier for new members to ramp up quickly. If you are unsure where to look for something, start by asking in the semester channel.

## 5. GitHub and codebase onboarding

The GitHub repository is the center of day-to-day development. In the docs folder of the Factor-Lake repository, the file CONTRIBUTING.md provides the step-by-step instructions for cloning the repository, creating a branch, and contributing code properly. That document should be your starting point before making any changes.

To understand the codebase, use the Trello card called "MkDocs Documentation Site" in the Reference list. That site explains the APIs and functions used throughout the repository and is the best available reference for understanding how pieces of the codebase fit together. In addition, the DOCS folder in the repository contains supporting documentation for specific tasks. For example, DEPLOYMENT.md explains how to deploy, maintain, and share the Streamlit app, while SUPABASE_SETUP.md walks you through configuring your Supabase environment.

### Recommended GitHub ramp-up sequence

1. Read docs/CONTRIBUTING.md fully before setting up your local environment.
2. Clone the repository and follow the documented branching workflow.
3. Use the MkDocs Documentation Site to understand relevant APIs and functions before editing code.
4. Check the DOCS folder for task-specific guidance such as deployment or Supabase setup.
5. Only begin implementation after the Trello story is clarified and moved into active sprint work.

## 6. Supabase usage

Supabase serves as the backend database for the project. Most team members will not need to work in it frequently. In practice, it is mainly touched when the project needs new data. In those cases, a member may create a new table in the project or populate it with the required data.

If your task involves local setup or environment configuration related to the database, use SUPABASE_SETUP.md in the repository documentation. That file should be treated as the standard reference for getting your Supabase environment working correctly.

## 7. How Trello is used

Trello is the team's operating board for planning, estimating, assigning, reviewing, and presenting work. The lists move from left to right and represent the lifecycle of a feature or story. A new member does not need to memorize everything immediately, but understanding the flow below will make the weekly meetings and sprint process much easier to follow.

| Order | List name | Meaning in practice |
|-------|-----------|-------------------|
| 1 | Retros | Used during the weekly meeting with the professor to reflect on the previous sprint: what worked, what did not work, and what should be done differently next time. The professor usually records the team's answers. |
| 2 | Reference | Stores important links and resources used by the team. Key cards include AWS credentials, MkDocs Documentation Site, Streamlit App, Code Repo, Streamlit Secrets, and ScrumPoker. |
| 3 | Freezer | Holds ideas or features that may be valuable later but are not a near-term priority. |
| 4 | Product Backlog | Contains features that are expected to be implemented in the near future. Many cards already include user stories written in the form "As a ... I want to ... so that I can ...". |
| 5 | Sprint Backlog | Contains the stories selected for the upcoming sprint after they have been clarified and estimated. These are the highest-priority items for the team. |
| 6 | Doing | A member moves a card here when they begin work so the rest of the team knows it is actively being handled and duplicated effort is avoided. |
| 7 | Awaiting PR Approval | The work is complete in the branch, and a pull request has been opened. Another member now reviews it to verify that the new feature works and does not break anything else. |
| 8 | Ready for Product Owner | Used after a PR approves and merges the new changes into main. These cards are ready to be demoed during the next meeting with the professor. |
| 9 | Done | Cards are moved here by the professor once the demo is accepted and the work is considered complete. |

## 8. Sprint planning, estimation, and implementation flow

During the weekly team meeting, members discuss which stories should be attempted in the next sprint. Stories are clarified first so ambiguities are resolved before implementation begins. Once a story is clear enough, the team uses ScrumPoker to estimate effort.

ScrumPoker estimates are not time estimates, they are effort estimates. When voting, members with unusually high or low estimates explain their reasoning. Members closer to the average also explain their thinking. The team then discusses the differences until it reaches a shared estimate. After enough stories are clarified and estimated, the team selects the ones that matter most, and are doable within the weeklong sprint based on team priorities, client needs from the Cayuga Fund, and/or guidance from the professor.

Once a member starts a story, they open it in "Doing". After implementation is complete, they open a pull request from their branch to main, describe the changes clearly, and move the card to "Awaiting PR Approval". A second member then reviews the pull request. If the change works as intended and does not negatively affect other parts of the product, the reviewer merges the change into main and moves the card to "Ready for Product Owner".

In the next meeting with the professor, all cards in "Ready for Product Owner" are demoed first. If the work solves the intended problem, the professor moves the card to "Done". If not, the card is moved back to "Sprint Backlog" to be attempted again in the next sprint.

## 9. What happens in the weekly meetings

| Meeting | Typical focus |
|---------|--------------|
| Weekly team meeting | The team clarifies stories, estimates them using ScrumPoker, decides which work belongs in the sprint, and aligns on who will take which cards. |
| Weekly meeting with the professor | The team demos completed work first, reviews retrospectives, discusses what should happen next in the project, and aligns on priority features or adjustments for the next sprint. |

## 10. What is expected from a new member

A new member is not expected to understand the entire project immediately. The goal is to learn the workflow quickly, ask questions early, and use the existing documentation before making changes. In practice, this means reading the repository documentation, using the MkDocs site to understand code that you touch, communicating on Slack when blocked, and keeping Trello updated whenever you begin or finish work on a story.

The most important habit is visibility. If you start working on a card, move it to "Doing." If you finish and open a pull request, move it to "Awaiting PR Approval." If something is unclear, ask in Slack instead of guessing. The project runs smoothly when documentation, communication, and board status all stay aligned.

## 11. Quick reference

| Need | Where to go |
|------|------------|
| How to clone, branch, and contribute | docs/CONTRIBUTING.md in the GitHub repository |
| How to understand the codebase | Trello "Reference" list → "MkDocs Documentation Site" |
| How to deploy or manage the Streamlit app | DEPLOYMENT.md in the repository documentation |
| How to configure Supabase locally | SUPABASE_SETUP.md in the repository documentation |
| Where links and team resources live | Trello "Reference" list |
| Where to ask questions | The semester Slack channel: fall-20XX or spring-20XX |

## 12. Final note

This project already has a working structure for code contribution, documentation, sprint planning, and communication. Your job as a new member is not to reinvent that structure, but to learn it, use it consistently, and contribute clearly within it. When in doubt, begin with the documentation, check Trello, and ask in Slack.
