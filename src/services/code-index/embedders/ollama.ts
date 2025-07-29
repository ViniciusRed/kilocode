import { ApiHandlerOptions } from "../../../shared/api"
import { EmbedderInfo, EmbeddingResponse, IEmbedder } from "../interfaces"
import { getModelQueryPrefix } from "../../../shared/embeddingModels"
import { MAX_ITEM_TOKENS } from "../constants"
import { t } from "../../../i18n"
import { withValidationErrorHandling, sanitizeErrorMessage } from "../shared/validation-helpers"
import { TelemetryService } from "@roo-code/telemetry"
import { TelemetryEventName } from "@roo-code/types"

// Adaptive timeout constants
const BASE_TIMEOUT_MS = 30000 // 30 seconds base timeout
const TIMEOUT_PER_TEXT_MS = 2000 // 2 seconds per text item
const MAX_TIMEOUT_MS = 300000 // 5 minutes maximum timeout
const MIN_TIMEOUT_MS = 15000 // 15 seconds minimum timeout
const VALIDATION_TIMEOUT_MS = 30000 // 30 seconds for validation requests

// Retry configuration
const MAX_RETRIES = 3
const INITIAL_RETRY_DELAY_MS = 1000
const MAX_RETRY_DELAY_MS = 10000

/**
 * Implements the IEmbedder interface using a local Ollama instance with robust timeout handling.
 */
export class CodeIndexOllamaEmbedder implements IEmbedder {
	private readonly baseUrl: string
	private readonly defaultModelId: string

	constructor(options: ApiHandlerOptions) {

		// Ensure ollamaBaseUrl and ollamaModelId exist on ApiHandlerOptions or add defaults
		let baseUrl = options.ollamaBaseUrl || "http://localhost:11434"

		// Normalize the baseUrl by removing all trailing slashes
		baseUrl = baseUrl.replace(/\/+$/, "")

		this.baseUrl = baseUrl

		this.defaultModelId = options.ollamaModelId || "nomic-embed-text:latest"
	}

	/**
	 * Calculate adaptive timeout based on the number of texts and their content
	 */
	private calculateAdaptiveTimeout(texts: string[]): number {
		// Calculate timeout based on number of texts and their average length
		const avgLength = texts.reduce((sum, text) => sum + text.length, 0) / texts.length
		const complexityFactor = Math.max(1, avgLength / 1000) // Longer texts need more time

		const calculatedTimeout = BASE_TIMEOUT_MS + texts.length * TIMEOUT_PER_TEXT_MS * complexityFactor

		// Clamp between min and max
		return Math.max(MIN_TIMEOUT_MS, Math.min(MAX_TIMEOUT_MS, calculatedTimeout))
	}

	/**
	 * Create a fetch request with adaptive timeout and retry logic
	 */
	private async fetchWithRetry(
		url: string,
		options: RequestInit,
		timeoutMs: number,
		maxRetries: number = MAX_RETRIES,
	): Promise<Response> {
		let lastError: Error | null = null

		for (let attempt = 0; attempt < maxRetries; attempt++) {
			try {
				// Create abort controller for this attempt
				const controller = new AbortController()
				let timeoutId: NodeJS.Timeout | null = null

				// Set up timeout only if timeoutMs > 0
				if (timeoutMs > 0) {
					timeoutId = setTimeout(() => {
						controller.abort()
					}, timeoutMs)
				}

				const requestOptions = {
					...options,
					signal: controller.signal,
				}

				try {
					const response = await fetch(url, requestOptions)

					// Clear timeout on successful response
					if (timeoutId) {
						clearTimeout(timeoutId)
					}

					return response
				} catch (fetchError) {
					// Clear timeout on error
					if (timeoutId) {
						clearTimeout(timeoutId)
					}
					throw fetchError
				}
			} catch (error: any) {
				lastError = error

				// Don't retry on certain errors
				if (error.name === "AbortError" && attempt < maxRetries - 1) {
					console.warn(`Request timeout on attempt ${attempt + 1}, retrying...`)
				} else if (error.message?.includes("fetch failed") || error.code === "ECONNREFUSED") {
					// Connection errors - don't retry
					throw error
				} else if (attempt === maxRetries - 1) {
					// Last attempt failed
					throw error
				}

				// Calculate retry delay with exponential backoff
				const baseDelay = INITIAL_RETRY_DELAY_MS * Math.pow(2, attempt)
				const delay = Math.min(baseDelay, MAX_RETRY_DELAY_MS)

				console.warn(`Request failed on attempt ${attempt + 1}, retrying in ${delay}ms...`)
				await new Promise((resolve) => setTimeout(resolve, delay))
			}
		}

		throw lastError || new Error("Max retries exceeded")
	}

	/**
	 * Process embeddings in chunks to avoid overwhelming the service
	 */
	private async processEmbeddingsInChunks(
		processedTexts: string[],
		modelToUse: string,
		chunkSize: number = 50, // Process in smaller chunks
	): Promise<number[][]> {
		const allEmbeddings: number[][] = []

		for (let i = 0; i < processedTexts.length; i += chunkSize) {
			const chunk = processedTexts.slice(i, i + chunkSize)
			const chunkTimeout = this.calculateAdaptiveTimeout(chunk)

			console.log(
				`Processing chunk ${Math.floor(i / chunkSize) + 1}/${Math.ceil(processedTexts.length / chunkSize)} (${chunk.length} items, timeout: ${chunkTimeout}ms)`,
			)

			const response = await this.fetchWithRetry(
				`${this.baseUrl}/api/embed`,
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
					body: JSON.stringify({
						model: modelToUse,
						input: chunk,
					}),
				},
				chunkTimeout,
			)

			if (!response.ok) {
				let errorBody = t("embeddings:ollama.couldNotReadErrorBody")
				try {
					errorBody = await response.text()
				} catch (e) {
					// Ignore error reading body
				}
				throw new Error(
					t("embeddings:ollama.requestFailed", {
						status: response.status,
						statusText: response.statusText,
						errorBody,
					}),
				)
			}

			const data = await response.json()
			const embeddings = data.embeddings

			if (!embeddings || !Array.isArray(embeddings)) {
				throw new Error(t("embeddings:ollama.invalidResponseStructure"))
			}

			allEmbeddings.push(...embeddings)
		}

		return allEmbeddings
	}

	/**
	 * Creates embeddings for the given texts using the specified Ollama model.
	 * @param texts - An array of strings to embed.
	 * @param model - Optional model ID to override the default.
	 * @returns A promise that resolves to an EmbeddingResponse containing the embeddings and usage data.
	 */
	async createEmbeddings(texts: string[], model?: string): Promise<EmbeddingResponse> {
		const modelToUse = model || this.defaultModelId

		// Apply model-specific query prefix if required
		const queryPrefix = getModelQueryPrefix("ollama", modelToUse)
		const processedTexts = queryPrefix
			? texts.map((text, index) => {
					// Prevent double-prefixing
					if (text.startsWith(queryPrefix)) {
						return text
					}
					const prefixedText = `${queryPrefix}${text}`
					const estimatedTokens = Math.ceil(prefixedText.length / 4)
					if (estimatedTokens > MAX_ITEM_TOKENS) {
						console.warn(
							t("embeddings:textWithPrefixExceedsTokenLimit", {
								index,
								estimatedTokens,
								maxTokens: MAX_ITEM_TOKENS,
							}),
						)
						return text
					}
					return prefixedText
				})
			: texts

		try {
			console.log(`Creating embeddings for ${processedTexts.length} texts`)

			let embeddings: number[][]

			// For large batches, process in chunks
			if (processedTexts.length > 100) {
				embeddings = await this.processEmbeddingsInChunks(processedTexts, modelToUse)
			} else {
				// For smaller batches, process all at once with adaptive timeout
				const adaptiveTimeout = this.calculateAdaptiveTimeout(processedTexts)
				console.log(`Using adaptive timeout: ${adaptiveTimeout}ms for ${processedTexts.length} texts`)

				const response = await this.fetchWithRetry(
					`${this.baseUrl}/api/embed`,
					{
						method: "POST",
						headers: {
							"Content-Type": "application/json",
						},
						body: JSON.stringify({
							model: modelToUse,
							input: processedTexts,
						}),
					},
					adaptiveTimeout,
				)

				if (!response.ok) {
					let errorBody = t("embeddings:ollama.couldNotReadErrorBody")
					try {
						errorBody = await response.text()
					} catch (e) {
						// Ignore error reading body
					}
					throw new Error(
						t("embeddings:ollama.requestFailed", {
							status: response.status,
							statusText: response.statusText,
							errorBody,
						}),
					)
				}

				const data = await response.json()
				embeddings = data.embeddings

				if (!embeddings || !Array.isArray(embeddings)) {
					throw new Error(t("embeddings:ollama.invalidResponseStructure"))
				}
			}

			console.log(`Successfully created ${embeddings.length} embeddings`)
			return {
				embeddings: embeddings,
			}
		} catch (error: any) {
			// Capture telemetry before reformatting the error
			TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
				error: sanitizeErrorMessage(error instanceof Error ? error.message : String(error)),
				stack: error instanceof Error ? sanitizeErrorMessage(error.stack || "") : undefined,
				location: "OllamaEmbedder:createEmbeddings",
				batchSize: texts.length,
			})

			// Log the original error for debugging purposes
			console.error("Ollama embedding failed:", error)

			// Handle specific error types with better messages
			if (error.name === "AbortError") {
				throw new Error(t("embeddings:validation.connectionFailed"))
			} else if (error.message?.includes("fetch failed") || error.code === "ECONNREFUSED") {
				throw new Error(t("embeddings:ollama.serviceNotRunning", { baseUrl: this.baseUrl }))
			} else if (error.code === "ENOTFOUND") {
				throw new Error(t("embeddings:ollama.hostNotFound", { baseUrl: this.baseUrl }))
			}

			// Re-throw a more specific error for the caller
			throw new Error(t("embeddings:ollama.embeddingFailed", { message: error.message }))
		}
	}

	/**
	 * Validates the Ollama embedder configuration by checking service availability and model existence
	 * @returns Promise resolving to validation result with success status and optional error message
	 */
	async validateConfiguration(): Promise<{ valid: boolean; error?: string }> {
		return withValidationErrorHandling(
			async () => {
				// First check if Ollama service is running by trying to list models
				const modelsUrl = `${this.baseUrl}/api/tags`

				const modelsResponse = await this.fetchWithRetry(
					modelsUrl,
					{
						method: "GET",
						headers: {
							"Content-Type": "application/json",
						},
					},
					VALIDATION_TIMEOUT_MS,
					2, // Fewer retries for validation
				)

				if (!modelsResponse.ok) {
					if (modelsResponse.status === 404) {
						return {
							valid: false,
							error: t("embeddings:ollama.serviceNotRunning", { baseUrl: this.baseUrl }),
						}
					}
					return {
						valid: false,
						error: t("embeddings:ollama.serviceUnavailable", {
							baseUrl: this.baseUrl,
							status: modelsResponse.status,
						}),
					}
				}

				// Check if the specific model exists
				const modelsData = await modelsResponse.json()
				const models = modelsData.models || []

				// Check both with and without :latest suffix
				const modelExists = models.some((m: any) => {
					const modelName = m.name || ""
					return (
						modelName === this.defaultModelId ||
						modelName === `${this.defaultModelId}:latest` ||
						modelName === this.defaultModelId.replace(":latest", "")
					)
				})

				if (!modelExists) {
					const availableModels = models.map((m: any) => m.name).join(", ")
					return {
						valid: false,
						error: t("embeddings:ollama.modelNotFound", {
							modelId: this.defaultModelId,
							availableModels,
						}),
					}
				}

				// Try a test embedding to ensure the model works for embeddings
				const testResponse = await this.fetchWithRetry(
					`${this.baseUrl}/api/embed`,
					{
						method: "POST",
						headers: {
							"Content-Type": "application/json",
						},
						body: JSON.stringify({
							model: this.defaultModelId,
							input: ["test"],
						}),
					},
					VALIDATION_TIMEOUT_MS,
					2, // Fewer retries for validation
				)

				if (!testResponse.ok) {
					return {
						valid: false,
						error: t("embeddings:ollama.modelNotEmbeddingCapable", { modelId: this.defaultModelId }),
					}
				}

				return { valid: true }
			},
			"ollama",
			{
				beforeStandardHandling: (error: any) => {
					// Handle Ollama-specific connection errors
					if (
						error?.message?.includes("fetch failed") ||
						error?.code === "ECONNREFUSED" ||
						error?.message?.includes("ECONNREFUSED")
					) {
						TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
							error: sanitizeErrorMessage(error instanceof Error ? error.message : String(error)),
							stack: error instanceof Error ? sanitizeErrorMessage(error.stack || "") : undefined,
							location: "OllamaEmbedder:validateConfiguration:connectionFailed",
						})
						return {
							valid: false,
							error: t("embeddings:ollama.serviceNotRunning", { baseUrl: this.baseUrl }),
						}
					} else if (error?.code === "ENOTFOUND" || error?.message?.includes("ENOTFOUND")) {
						TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
							error: sanitizeErrorMessage(error instanceof Error ? error.message : String(error)),
							stack: error instanceof Error ? sanitizeErrorMessage(error.stack || "") : undefined,
							location: "OllamaEmbedder:validateConfiguration:hostNotFound",
						})
						return {
							valid: false,
							error: t("embeddings:ollama.hostNotFound", { baseUrl: this.baseUrl }),
						}
					} else if (error?.name === "AbortError") {
						TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
							error: sanitizeErrorMessage(error instanceof Error ? error.message : String(error)),
							stack: error instanceof Error ? sanitizeErrorMessage(error.stack || "") : undefined,
							location: "OllamaEmbedder:validateConfiguration:timeout",
						})
						return {
							valid: false,
							error: t("embeddings:validation.connectionFailed"),
						}
					}
					return undefined
				},
			},
		)
	}

	get embedderInfo(): EmbedderInfo {
		return {
			name: "ollama",
		}
	}
}
