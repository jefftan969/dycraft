---
name: save_context
description: Write a detailed context summary before pausing, switching tasks, or moving to a new context window.
---
# Skill: Summarize Context & Save

## Purpose
You have been working on the task described above but have not yet completed it. Your task is to write a detailed context summary of the conversation so far, that will allow you (or another instance of yourself) to resume work efficiently in a future context window where the conversation history will be replaced with this summary. This summary should be thorough in capturing the user's explicit requests, your previous actions, technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context. The context summary should be saved to a temporary file (`./context_summary.md`) to be read in future sessions. If this conversation itself has been resumed from a previous context summary saved on disk, include the entirety of that previous context summary in the current one. If any part of this session involved producing a summary, report, document, or analysis for the user, make sure to include the entirety of those artifacts in context_summary.md.

## Instructions
Analyze the current conversation, paying attention to decisions, file changes, project goals, and the motivation and impact of each command. Then, draft the complete context summary in Markdown, including:
1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail. Focus on the success criteria and any clarifications or constraints they specified.
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Current Work: Key work completed and key decisions made, including the motivation, history, and impact of each command run. Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
5. Errors and Fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
6. Problem Solving: Discuss any technical constraints or requirements that have been uncovered. Discuss any decisions made and their rationale. Document problems solved and any ongoing troubleshooting efforts.
7. All user messages: List ALL actual user messages — the text the user themselves wrote, excluding tool results and other automated turns. These are critical for understanding the user's feedback and changing intent.
8. All user artifacts: List ALL substantive technical deliverables you produced for the user as part of this conversation — reports, analyses, summaries, documents, plans, design docs, code explanations, reviews, or recommendations — whether saved to a file or delivered inline in the chat. Include the full text of each, verbatim. If the user engaged in an interactive design dialogue, include the full transcript of that discussion — all queries and responses — concentrating on the substantive technical content; you may drop purely non-technical boilerplate (acknowledgements, status updates, tool-call narration).
9. Pending Tasks: Outline any high-level pending tasks that you have explicitly been asked to work on. Discuss any blockers or open questions to resolve. Use priority order if multiple steps remain.
10. Next Steps: List the detailed specific actions that you will take as the next step related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first. If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.
11. Other Notes: Discuss any domain-specific details that aren't obvious, and any promises made to the user. Be concise but complete: err on the side of including information that would prevent duplicate work or repeated mistakes. Write in a way that enables immediate resumption of the task.

Here's an example of how your output should be structured:

1. Primary Request and Intent: [Detailed description]

2. Key Technical Concepts:
- [Concept 1]
- [Concept 2]
- [...]

3. Files and Code Sections:
- [File Name 1]
  - [Summary of why this file is important]
  - [Summary of the changes made to this file, if any]
  - [Important Code Snippet]
- [File Name 2]
  - [Important Code Snippet]
- [...]

4. Current Work: [Precise description of current work]

5. Errors and Fixes:
- [Detailed description of error 1]:
  - [How you fixed the error]
  - [User feedback on the error if any]
- [...]

6. Problem Solving: [Description of solved problems and ongoing troubleshooting]

7. All user messages:
- [Verbatim user message — the exact text the user wrote, not a tool result]
- [...]

8. All user artifacts:
- [Full verbatim text of a user artifact]
- [...]

9. Pending Tasks:
- [Task 1]
- [Task 2]
- [...]

10. Next Steps: [Next steps to take]

11. Other Notes: [Additional useful context to preserve]

Please provide the context summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

## When to use
- Before starting a new task within the same project.
- When the conversation is getting too long.
- When you want to pause and resume later.

## DyCraft context files

- Inside a DyCraft harness run, write `$DYCRAFT_RUN_DIR/context_summary.md` instead of `./context_summary.md`.
- Begin with a descriptive session title, date, and relevant branch, HEAD, and commit state.
- If resuming from a previous summary, carry its still-relevant facts into the current sections and reproduce it verbatim in an appendix after section 11.
- Under Errors and Fixes, record repairs required by checkers, verifiers, or reviewers, including what was rejected and why.
