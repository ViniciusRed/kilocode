import { Anthropic } from "@anthropic-ai/sdk"
import { parseMentions } from "./index"
import { UrlContentFetcher } from "../../services/browser/UrlContentFetcher"
import { FileContextTracker } from "../context-tracking/FileContextTracker"

import { GlobalFileNames } from "../../shared/globalFileNames"
import { ensureLocalKilorulesDirExists } from "../context/instructions/kilo-rules"
import { parseKiloSlashCommands } from "../slash-commands/kilo"
import { refreshWorkflowToggles } from "../context/instructions/workflows" // kilocode_change

import * as vscode from "vscode" // kilocode_change

// This function is a duplicate of processUserContentMentions, but it adds a check for the newrules command
// and processes Kilo-specific slash commands. It should be merged with processUserContentMentions in the future.
export async function processKiloUserContentMentions({
	context, // kilocode_change
	userContent,
	cwd,
	urlContentFetcher,
	fileContextTracker,
	rooIgnoreController,
	showRooIgnoredFiles = true,
	includeDiagnosticMessages = true,
	maxDiagnosticMessages = 50,
	maxReadFileLine,
}: {
	context: vscode.ExtensionContext // kilocode_change
	userContent: Anthropic.Messages.ContentBlockParam[]
	cwd: string
	urlContentFetcher: UrlContentFetcher
	fileContextTracker: FileContextTracker
	rooIgnoreController?: any
	showRooIgnoredFiles: boolean
	includeDiagnosticMessages?: boolean
	maxDiagnosticMessages?: number
	maxReadFileLine?: number
}): Promise<[Anthropic.Messages.ContentBlockParam[], boolean]> {
	// Track if we need to check kilorules file
	let needsRulesFileCheck = false

	/**
	 * Process mentions in user content, specifically within task and feedback tags
	 */
	const processUserContentMentions = async () => {
		// Process userContent array, which contains various block types:
		// TextBlockParam, ImageBlockParam, ToolUseBlockParam, and ToolResultBlockParam.
		// We need to apply parseMentions() to:
		// 1. All TextBlockParam's text (first user message with task)
		// 2. ToolResultBlockParam's content/context text arrays if it contains
		// "<feedback>" (see formatToolDeniedFeedback, attemptCompletion,
		// executeCommand, and consecutiveMistakeCount >= 3) or "<answer>"
		// (see askFollowupQuestion), we place all user generated content in
		// these tags so they can effectively be used as markers for when we
		// should parse mentions).

		const { localWorkflowToggles, globalWorkflowToggles } = await refreshWorkflowToggles(context, cwd) // kilocode_change

		return await Promise.all(
			userContent.map(async (block) => {
				const shouldProcessMentions = (text: string) => text.includes("<task>") || text.includes("<feedback>")

				if (block.type === "text") {
					if (shouldProcessMentions(block.text)) {
						// kilocode_change begin: pull slash commands from Cline
						const parsedText = await parseMentions(
							block.text,
							cwd,
							urlContentFetcher,
							fileContextTracker,
							rooIgnoreController,
							showRooIgnoredFiles,
							includeDiagnosticMessages,
							maxDiagnosticMessages,
							maxReadFileLine,
						)

						// when parsing slash commands, we still want to allow the user to provide their desired context
						const { processedText, needsRulesFileCheck: needsCheck } = await parseKiloSlashCommands(
							parsedText,
							localWorkflowToggles, // kilocode_change
							globalWorkflowToggles, // kilocode_change
						)

						if (needsCheck) {
							needsRulesFileCheck = true
						}

						return {
							...block,
							text: processedText,
						}
						// kilocode_change end
					}

					return block
				} else if (block.type === "tool_result") {
					if (typeof block.content === "string") {
						if (shouldProcessMentions(block.content)) {
							return {
								...block,
								content: await parseMentions(
									block.content,
									cwd,
									urlContentFetcher,
									fileContextTracker,
									rooIgnoreController,
									showRooIgnoredFiles,
									includeDiagnosticMessages,
									maxDiagnosticMessages,
									maxReadFileLine,
								),
							}
						}

						return block
					} else if (Array.isArray(block.content)) {
						const parsedContent = await Promise.all(
							block.content.map(async (contentBlock) => {
								if (contentBlock.type === "text" && shouldProcessMentions(contentBlock.text)) {
									return {
										...contentBlock,
										text: await parseMentions(
											contentBlock.text,
											cwd,
											urlContentFetcher,
											fileContextTracker,
											rooIgnoreController,
											showRooIgnoredFiles,
											includeDiagnosticMessages,
											maxDiagnosticMessages,
											maxReadFileLine,
										),
									}
								}

								return contentBlock
							}),
						)

						return { ...block, content: parsedContent }
					}

					return block
				}

				return block
			}),
		)
	}

	const processedUserContent = await processUserContentMentions()

	let kilorulesError = false
	if (needsRulesFileCheck) {
		kilorulesError = await ensureLocalKilorulesDirExists(cwd, GlobalFileNames.kiloRules)
	}
	return [processedUserContent, kilorulesError]
}
