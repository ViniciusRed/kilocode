// kilocode_change: Morph fast apply -- file added

import path from "path"
import { promises as fs } from "fs"
import OpenAI from "openai"
import { Ollama } from "ollama"

import { Task } from "../task/Task"
import { formatResponse } from "../prompts/responses"
import { ToolUse, AskApproval, HandleError, PushToolResult, RemoveClosingTag } from "../../shared/tools"
import { fileExistsAtPath } from "../../utils/fs"
import { getReadablePath } from "../../utils/path"
import { Experiments, ProviderSettings } from "@roo-code/types"
import { getKiloBaseUriFromToken } from "../../shared/kilocode/token"
import { DEFAULT_HEADERS } from "../../api/providers/constants"
import { TelemetryService } from "@roo-code/telemetry"

async function validateParams(
	cline: Task,
	targetFile: string | undefined,
	instructions: string | undefined,
	codeEdit: string | undefined,
	pushToolResult: PushToolResult,
): Promise<boolean> {
	if (!targetFile) {
		cline.consecutiveMistakeCount++
		cline.recordToolError("edit_file")
		pushToolResult(await cline.sayAndCreateMissingParamError("edit_file", "target_file"))
		return false
	}

	if (!instructions) {
		cline.consecutiveMistakeCount++
		cline.recordToolError("edit_file")
		pushToolResult(await cline.sayAndCreateMissingParamError("edit_file", "instructions"))
		return false
	}

	if (codeEdit === undefined) {
		cline.consecutiveMistakeCount++
		cline.recordToolError("edit_file")
		pushToolResult(await cline.sayAndCreateMissingParamError("edit_file", "code_edit"))
		return false
	}

	return true
}

export async function editFileTool(
	cline: Task,
	block: ToolUse,
	askApproval: AskApproval,
	handleError: HandleError,
	pushToolResult: PushToolResult,
	removeClosingTag: RemoveClosingTag,
): Promise<void> {
	const target_file: string | undefined = block.params.target_file
	const instructions: string | undefined = block.params.instructions
	const code_edit: string | undefined = block.params.code_edit

	try {
		// Handle partial tool use
		if (block.partial) {
			const partialMessageProps = {
				tool: "editFile" as const,
				path: getReadablePath(cline.cwd, removeClosingTag("target_file", target_file)),
				instructions: removeClosingTag("instructions", instructions),
				codeEdit: removeClosingTag("code_edit", code_edit),
			}
			await cline.ask("tool", JSON.stringify(partialMessageProps), block.partial).catch(() => {
				// Roo tools ignore exceptions as well here
			})
			return
		}

		// Validate required parameters
		if (!(await validateParams(cline, target_file, instructions, code_edit, pushToolResult))) {
			return
		}

		// At this point we know all parameters are defined, so we can safely cast them
		const targetFile = target_file as string
		const editInstructions = instructions as string
		const editCode = code_edit as string

		// Validate and resolve the file path
		const absolutePath = path.resolve(cline.cwd, targetFile)
		const relPath = getReadablePath(cline.cwd, absolutePath)

		// Check if file exists
		if (!(await fileExistsAtPath(absolutePath))) {
			cline.consecutiveMistakeCount++
			cline.recordToolError("edit_file")
			pushToolResult(
				formatResponse.toolError(
					`The file ${relPath} does not exist. Use write_to_file to create new files, or make sure the file path is correct.`,
				),
			)
			return
		}

		// Check if file access is allowed
		const accessAllowed = cline.rooIgnoreController?.validateAccess(relPath)
		if (!accessAllowed) {
			await cline.say("rooignore_error", relPath)
			pushToolResult(formatResponse.rooIgnoreError(relPath))
			return
		}

		// Read the original file content
		const originalContent = await fs.readFile(absolutePath, "utf-8")

		// Get the current API configuration
		const provider = cline.providerRef.deref()
		if (!provider) {
			cline.consecutiveMistakeCount++
			cline.recordToolError("edit_file")
			pushToolResult(formatResponse.toolError("No API provider available"))
			return
		}

		const state = await provider.getState()
		const fastApplyMethod = (state.apiConfiguration as any).fastApplyMethod || "morph"

		let newContent: string
		let applyResult: any

		if (fastApplyMethod === "morph") {
			// Try Morph first
			applyResult = await applyMorphEdit(originalContent, editInstructions, editCode, cline)
			if (!applyResult.success) {
				// Try Ollama as fallback if Morph fails
				applyResult = await applyOllamaFastEdit(originalContent, editInstructions, editCode, cline)
			}
		} else {
			// Try Ollama first
			applyResult = await applyOllamaFastEdit(originalContent, editInstructions, editCode, cline)
			if (!applyResult.success) {
				// Try Morph as fallback if Ollama fails
				applyResult = await applyMorphEdit(originalContent, editInstructions, editCode, cline)
			}
		}

		if (!applyResult.success) {
			cline.consecutiveMistakeCount++
			cline.recordToolError("edit_file")

			const errorMethod = fastApplyMethod === "morph" ? "Morph" : "Ollama Fast Apply"
			pushToolResult(
				formatResponse.toolError(
					`Failed to apply edit using ${errorMethod}: ${applyResult.error}. Consider using apply_diff tool instead.`,
				),
			)
			return
		}

		newContent = applyResult.result!

		// Show the diff and ask for approval
		cline.diffViewProvider.editType = "modify"
		await cline.diffViewProvider.open(relPath)

		// Stream the content to show the diff
		await cline.diffViewProvider.update(newContent, true)
		cline.diffViewProvider.scrollToFirstDiff()

		// Ask for user approval
		const approved = await askApproval(
			"tool",
			JSON.stringify({
				tool: "editedExistingFile",
				path: relPath,
				isProtected: cline.rooProtectedController?.isWriteProtected(relPath) || false,
				instructions: editInstructions,
			}),
			undefined,
			cline.rooProtectedController?.isWriteProtected(relPath) || false,
		)

		if (!approved) {
			await cline.diffViewProvider.revertChanges()
			pushToolResult(formatResponse.toolResult("Edit cancelled by user."))
			return
		}

		// Apply the changes
		await cline.diffViewProvider.saveChanges()

		// Track file context
		await cline.fileContextTracker.trackFileContext(relPath, "roo_edited")
		cline.didEditFile = true
		cline.consecutiveMistakeCount = 0

		// Get the formatted response message
		const message = await cline.diffViewProvider.pushToolWriteResult(cline, cline.cwd, false)
		pushToolResult(message)

		await cline.diffViewProvider.reset()
	} catch (error) {
		TelemetryService.instance.captureException(error, { context: "editFileTool" })
		await handleError("editing file with Morph", error as Error)
		await cline.diffViewProvider.reset()
	}
}

interface MorphApplyResult {
	success: boolean
	result?: string
	error?: string
}

interface OllamaFastApplyResult {
	success: boolean
	result?: string
	error?: string
}
async function applyMorphEdit(
	originalContent: string,
	instructions: string,
	codeEdit: string,
	cline: Task,
): Promise<MorphApplyResult> {
	try {
		// Get the current API configuration
		const provider = cline.providerRef.deref()
		if (!provider) {
			return { success: false, error: "No API provider available" }
		}

		const state = await provider.getState()

		// Check if user has Morph enabled via OpenRouter or direct API
		const morphConfig = await getMorphConfiguration(state.experiments, state.apiConfiguration)
		if (!morphConfig.available) {
			return { success: false, error: morphConfig.error || "Morph is not available" }
		}

		// Create OpenAI client for Morph API
		const client = new OpenAI({
			apiKey: morphConfig.apiKey,
			baseURL: morphConfig.baseUrl,
			defaultHeaders: {
				"X-KiloCode-TaskId": cline.taskId,
				...DEFAULT_HEADERS,
			},
		})

		// Apply the edit using Morph's format
		const prompt = `<instructions>${instructions}</instructions>\n<code>${originalContent}</code>\n<update>${codeEdit}</update>`

		const response = await client.chat.completions.create(
			{
				model: morphConfig.model!,
				messages: [
					{
						role: "user",
						content: prompt,
					},
				],
			},
			{
				timeout: 30000, // 30 second timeout
			},
		)

		const mergedCode = response.choices[0]?.message?.content
		if (!mergedCode) {
			return { success: false, error: "Morph API returned empty response" }
		}

		return { success: true, result: mergedCode }
	} catch (error) {
		TelemetryService.instance.captureException(error, { context: "applyMorphEdit" })
		return {
			success: false,
			error: error instanceof Error ? error.message : "Unknown error occurred",
		}
	}
}

async function applyOllamaFastEdit(
	originalContent: string,
	instructions: string,
	codeEdit: string,
	cline: Task,
): Promise<OllamaFastApplyResult> {
	try {
		// Get the current API configuration
		const provider = cline.providerRef.deref()
		if (!provider) {
			return { success: false, error: "No API provider available" }
		}

		const state = await provider.getState()
		const apiConfig = state.apiConfiguration

		// Check if Ollama is configured and FastApply is enabled
		if (apiConfig.apiProvider !== "ollama") {
			return { success: false, error: "Ollama is not configured as the API provider" }
		}

		// Create Ollama client
		const ollamaClient = new Ollama({
			host: apiConfig.ollamaBaseUrl || "http://localhost:11434",
		})

		// Get the selected FastApply model from configuration
		const fastApplyModel = (apiConfig as any).ollamaFastApplyModelId || "Kortix/FastApply-1.5B-v1.0"

		// Prepare the prompt following the FastApply format
		const prompt = `You are a coding assistant that helps merge code updates, ensuring every modification is fully integrated.

Merge all changes from the <update> snippet into the <code> below.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.

<code>${originalContent}</code>

<update>${codeEdit}</update>

Provide the complete updated code.`

		// Apply the edit using Ollama FastApply model
		const response = await ollamaClient.chat({
			model: fastApplyModel, // Use the selected model from configuration
			messages: [
				{
					role: "user",
					content: prompt,
				},
			],
			options: {
				temperature: 0,
				num_predict: 8192,
			},
		})

		const mergedCode = response.message.content
		if (!mergedCode) {
			return { success: false, error: "Ollama Fast Apply returned empty response" }
		}

		// Extract the updated code from the response
		const updatedCodeMatch = mergedCode.match(/<updated-code>([\s\S]*?)<\/updated-code>/)
		if (!updatedCodeMatch) {
			return { success: false, error: "Could not extract updated code from Ollama response" }
		}

		return { success: true, result: updatedCodeMatch[1].trim() }
	} catch (error) {
		TelemetryService.instance.captureException(error, { context: "applyOllamaFastEdit" })
		return {
			success: false,
			error: error instanceof Error ? error.message : "Unknown error occurred",
		}
	}
}

interface MorphConfiguration {
	available: boolean
	apiKey?: string
	baseUrl?: string
	model?: string
	error?: string
}

async function getMorphConfiguration(
	experiments: Experiments,
	apiConfig: ProviderSettings,
): Promise<MorphConfiguration> {
	// Check if Fast Apply is enabled in API configuration
	if (experiments.fastApply !== true) {
		return {
			available: false,
			error: "Fast Apply is disabled. Enable it in API Options > Enable Fast Apply",
		}
	}

	// If user has direct Morph API key, use it
	if (apiConfig.morphApiKey) {
		return {
			available: true,
			apiKey: apiConfig.morphApiKey,
			baseUrl: "https://api.morphllm.com/v1",
			model: "auto",
		}
	}

	if (apiConfig.apiProvider === "kilocode") {
		const token = apiConfig.kilocodeToken
		if (!token) {
			return { available: false, error: "No KiloCode token available to use Morph" }
		}
		return {
			available: true,
			apiKey: token,
			baseUrl: `${getKiloBaseUriFromToken(token)}/api/openrouter/`,
			model: "morph/morph-v3-large", // Morph model via OpenRouter
		}
	}

	// If user is using OpenRouter as their provider, use Morph through OpenRouter
	if (apiConfig.apiProvider === "openrouter") {
		const token = apiConfig.openRouterApiKey
		if (!token) {
			return { available: false, error: "No Openrouter api token available to use Morph" }
		}
		return {
			available: true,
			apiKey: token,
			baseUrl: apiConfig.openRouterBaseUrl || "https://openrouter.ai/api/v1",
			model: "morph/morph-v3-large", // Morph model via OpenRouter
		}
	}

	return {
		available: false,
		error: "Morph is enabled but not configured. Either set a Morph API key in API Options or use OpenRouter with Morph access.",
	}
}
