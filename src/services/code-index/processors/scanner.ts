import { listFiles } from "../../glob/list-files"
import { Ignore } from "ignore"
import { RooIgnoreController } from "../../../core/ignore/RooIgnoreController"
import { stat } from "fs/promises"
import * as path from "path"
import { generateNormalizedAbsolutePath, generateRelativeFilePath } from "../shared/get-relative-path"
import { getWorkspacePathForContext } from "../../../utils/path"
import { scannerExtensions } from "../shared/supported-extensions"
import * as vscode from "vscode"
import { CodeBlock, ICodeParser, IEmbedder, IVectorStore, IDirectoryScanner } from "../interfaces"
import { createHash } from "crypto"
import { v5 as uuidv5 } from "uuid"
import pLimit from "p-limit"
import { Mutex } from "async-mutex"
import { CacheManager } from "../cache-manager"
import { t } from "../../../i18n"
import {
	QDRANT_CODE_BLOCK_NAMESPACE,
	MAX_FILE_SIZE_BYTES,
	MAX_LIST_FILES_LIMIT_CODE_INDEX,
	BATCH_SEGMENT_THRESHOLD,
	MAX_BATCH_RETRIES,
	INITIAL_RETRY_DELAY_MS,
	PARSING_CONCURRENCY,
	BATCH_PROCESSING_CONCURRENCY,
	MAX_PENDING_BATCHES,
} from "../constants"
import { isPathInIgnoredDirectory } from "../../glob/ignore-utils"
import { TelemetryService } from "@roo-code/telemetry"
import { TelemetryEventName } from "@roo-code/types"
import { sanitizeErrorMessage } from "../shared/validation-helpers"

// Enhanced batch processing configuration
const ADAPTIVE_BATCH_SIZE_MIN = 10
const ADAPTIVE_BATCH_SIZE_MAX = 100
const EXPONENTIAL_BACKOFF_MAX_MS = 30000 // 30 seconds max delay

export class DirectoryScanner implements IDirectoryScanner {
	constructor(
		private readonly embedder: IEmbedder,
		private readonly qdrantClient: IVectorStore,
		private readonly codeParser: ICodeParser,
		private readonly cacheManager: CacheManager,
		private readonly ignoreInstance: Ignore,
	) {}

	/**
	 * Calculate adaptive batch size based on current system load and previous performance
	 */
	private calculateAdaptiveBatchSize(totalItems: number, averageProcessingTime: number = 0): number {
		// Start with a base size
		let batchSize = BATCH_SEGMENT_THRESHOLD

		// Adjust based on total items
		if (totalItems > 1000) {
			batchSize = Math.max(ADAPTIVE_BATCH_SIZE_MIN, Math.floor(batchSize * 0.7))
		} else if (totalItems < 100) {
			batchSize = Math.min(ADAPTIVE_BATCH_SIZE_MAX, Math.floor(batchSize * 1.3))
		}

		// Adjust based on processing time (if available)
		if (averageProcessingTime > 30000) {
			// If previous batches took > 30s
			batchSize = Math.max(ADAPTIVE_BATCH_SIZE_MIN, Math.floor(batchSize * 0.5))
		} else if (averageProcessingTime < 10000) {
			// If previous batches took < 10s
			batchSize = Math.min(ADAPTIVE_BATCH_SIZE_MAX, Math.floor(batchSize * 1.2))
		}

		return Math.max(ADAPTIVE_BATCH_SIZE_MIN, Math.min(ADAPTIVE_BATCH_SIZE_MAX, batchSize))
	}

	/**
	 * Recursively scans a directory for code blocks in supported files.
	 */
	public async scanDirectory(
		directory: string,
		onError?: (error: Error) => void,
		onBlocksIndexed?: (indexedCount: number) => void,
		onFileParsed?: (fileBlockCount: number) => void,
	): Promise<{ stats: { processed: number; skipped: number }; totalBlockCount: number }> {
		const directoryPath = directory
		const scanWorkspace = getWorkspacePathForContext(directoryPath)

		// Get all files recursively
		const [allPaths, _] = await listFiles(directoryPath, true, MAX_LIST_FILES_LIMIT_CODE_INDEX)
		const filePaths = allPaths.filter((p) => !p.endsWith("/"))

		// Initialize RooIgnoreController
		const ignoreController = new RooIgnoreController(directoryPath)
		await ignoreController.initialize()

		// Filter paths
		const allowedPaths = ignoreController.filterPaths(filePaths)
		const supportedPaths = allowedPaths.filter((filePath) => {
			const ext = path.extname(filePath).toLowerCase()
			const relativeFilePath = generateRelativeFilePath(filePath, scanWorkspace)

			if (isPathInIgnoredDirectory(filePath)) {
				return false
			}

			return scannerExtensions.includes(ext) && !this.ignoreInstance.ignores(relativeFilePath)
		})

		// Initialize tracking variables
		const processedFiles = new Set<string>()
		let processedCount = 0
		let skippedCount = 0
		let totalBlockCount = 0

		// Performance tracking for adaptive batching
		const batchPerformanceHistory: number[] = []

		// Initialize parallel processing tools
		const parseLimiter = pLimit(PARSING_CONCURRENCY)
		const batchLimiter = pLimit(BATCH_PROCESSING_CONCURRENCY)
		const mutex = new Mutex()

		// Shared batch accumulators
		let currentBatchBlocks: CodeBlock[] = []
		let currentBatchTexts: string[] = []
		let currentBatchFileInfos: { filePath: string; fileHash: string; isNew: boolean }[] = []
		const activeBatchPromises = new Set<Promise<void>>()
		let pendingBatchCount = 0

		// Process all files in parallel with concurrency control
		const parsePromises = supportedPaths.map((filePath) =>
			parseLimiter(async () => {
				try {
					// Check file size
					const stats = await stat(filePath)
					if (stats.size > MAX_FILE_SIZE_BYTES) {
						skippedCount++
						return
					}

					// Read file content
					const content = await vscode.workspace.fs
						.readFile(vscode.Uri.file(filePath))
						.then((buffer) => Buffer.from(buffer).toString("utf-8"))

					// Calculate current hash
					const currentFileHash = createHash("sha256").update(content).digest("hex")
					processedFiles.add(filePath)

					// Check against cache
					const cachedFileHash = this.cacheManager.getHash(filePath)
					const isNewFile = !cachedFileHash
					if (cachedFileHash === currentFileHash) {
						skippedCount++
						return
					}

					// Parse file
					const blocks = await this.codeParser.parseFile(filePath, { content, fileHash: currentFileHash })
					const fileBlockCount = blocks.length
					onFileParsed?.(fileBlockCount)
					processedCount++

					// Process embeddings if configured
					if (this.embedder && this.qdrantClient && blocks.length > 0) {
						let addedBlocksFromFile = false
						for (const block of blocks) {
							const trimmedContent = block.content.trim()
							if (trimmedContent) {
								const release = await mutex.acquire()
								try {
									currentBatchBlocks.push(block)
									currentBatchTexts.push(trimmedContent)
									addedBlocksFromFile = true

									// Calculate adaptive batch size
									const avgProcessingTime =
										batchPerformanceHistory.length > 0
											? batchPerformanceHistory.reduce((a, b) => a + b, 0) /
												batchPerformanceHistory.length
											: 0

									const adaptiveBatchSize = this.calculateAdaptiveBatchSize(
										supportedPaths.length,
										avgProcessingTime,
									)

									// Check if adaptive batch threshold is met
									if (currentBatchBlocks.length >= adaptiveBatchSize) {

										const batchBlocks = [...currentBatchBlocks]
										const batchTexts = [...currentBatchTexts]
										const batchFileInfos = [...currentBatchFileInfos]
										currentBatchBlocks = []
										currentBatchTexts = []
										currentBatchFileInfos = []


										// Queue batch processing with performance tracking
										const batchPromise = batchLimiter(async () => {
											const startTime = Date.now()
											await this.processRobustBatch(

												batchBlocks,
												batchTexts,
												batchFileInfos,
												scanWorkspace,
												onError,
												onBlocksIndexed,
											)
											const processingTime = Date.now() - startTime

											// Update performance history (keep last 10 measurements)
											batchPerformanceHistory.push(processingTime)
											if (batchPerformanceHistory.length > 10) {
												batchPerformanceHistory.shift()
											}
										})

										activeBatchPromises.add(batchPromise)
										batchPromise.finally(() => {
											activeBatchPromises.delete(batchPromise)
											pendingBatchCount--
										})
									}
								} finally {
									release()
								}
							}
						}

						// Add file info once per file
						if (addedBlocksFromFile) {
							const release = await mutex.acquire()
							try {
								totalBlockCount += fileBlockCount
								currentBatchFileInfos.push({
									filePath,
									fileHash: currentFileHash,
									isNew: isNewFile,
								})
							} finally {
								release()
							}
						}
					} else {
						await this.cacheManager.updateHash(filePath, currentFileHash)
					}
				} catch (error) {
					console.error(`Error processing file ${filePath} in workspace ${scanWorkspace}:`, error)
					TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
						error: sanitizeErrorMessage(error instanceof Error ? error.message : String(error)),
						stack: error instanceof Error ? sanitizeErrorMessage(error.stack || "") : undefined,
						location: "scanDirectory:processFile",
					})
					if (onError) {
						onError(
							error instanceof Error
								? new Error(`${error.message} (Workspace: ${scanWorkspace}, File: ${filePath})`)
								: new Error(
										t("embeddings:scanner.unknownErrorProcessingFile", { filePath }) +
											` (Workspace: ${scanWorkspace})`,
									),
						)
					}
				}
			}),
		)

		// Wait for all parsing to complete
		await Promise.all(parsePromises)

		// Process any remaining items in batch
		if (currentBatchBlocks.length > 0) {
			const release = await mutex.acquire()
			try {
				const batchBlocks = [...currentBatchBlocks]
				const batchTexts = [...currentBatchTexts]
				const batchFileInfos = [...currentBatchFileInfos]
				currentBatchBlocks = []
				currentBatchTexts = []
				currentBatchFileInfos = []



				const batchPromise = batchLimiter(() =>
					this.processRobustBatch(
						batchBlocks,
						batchTexts,
						batchFileInfos,
						scanWorkspace,
						onError,
						onBlocksIndexed,
					),
				)
				activeBatchPromises.add(batchPromise)
				batchPromise.finally(() => {
					activeBatchPromises.delete(batchPromise)
					pendingBatchCount--
				})
			} finally {
				release()
			}
		}

		// Wait for all batch processing to complete
		await Promise.all(activeBatchPromises)

		// Handle deleted files
		const oldHashes = this.cacheManager.getAllHashes()
		for (const cachedFilePath of Object.keys(oldHashes)) {
			if (!processedFiles.has(cachedFilePath)) {
				if (this.qdrantClient) {
					try {
						await this.qdrantClient.deletePointsByFilePath(cachedFilePath)
						await this.cacheManager.deleteHash(cachedFilePath)
					} catch (error) {
						console.error(
							`[DirectoryScanner] Failed to delete points for ${cachedFilePath} in workspace ${scanWorkspace}:`,
							error,
						)
						TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
							error: sanitizeErrorMessage(error instanceof Error ? error.message : String(error)),
							stack: error instanceof Error ? sanitizeErrorMessage(error.stack || "") : undefined,
							location: "scanDirectory:deleteRemovedFiles",
						})
						if (onError) {
							onError(
								error instanceof Error
									? new Error(
											`${error.message} (Workspace: ${scanWorkspace}, File: ${cachedFilePath})`,
										)
									: new Error(
											t("embeddings:scanner.unknownErrorDeletingPoints", {
												filePath: cachedFilePath,
											}) + ` (Workspace: ${scanWorkspace})`,
										),
							)
						}
					}
				}
			}
		}

		return {
			stats: {
				processed: processedCount,
				skipped: skippedCount,
			},
			totalBlockCount,
		}
	}

	/**
	 * Enhanced batch processing with exponential backoff and circuit breaker pattern
	 */
	private async processRobustBatch(
		batchBlocks: CodeBlock[],
		batchTexts: string[],
		batchFileInfos: { filePath: string; fileHash: string; isNew: boolean }[],
		scanWorkspace: string,
		onError?: (error: Error) => void,
		onBlocksIndexed?: (indexedCount: number) => void,
	): Promise<void> {
		if (batchBlocks.length === 0) return

		let attempts = 0
		let success = false
		let lastError: Error | null = null
		let currentBatchSize = batchBlocks.length

		console.log(`Processing batch with ${currentBatchSize} blocks`)

		while (attempts < MAX_BATCH_RETRIES && !success) {
			attempts++

			try {
				// --- Deletion Step ---
				const uniqueFilePaths = [
					...new Set(batchFileInfos.filter((info) => !info.isNew).map((info) => info.filePath)),
				]

				if (uniqueFilePaths.length > 0) {
					console.log(`Deleting existing points for ${uniqueFilePaths.length} files`)
					try {
						await this.qdrantClient.deletePointsByMultipleFilePaths(uniqueFilePaths)
					} catch (deleteError) {
						console.error(
							`[DirectoryScanner] Failed to delete points for ${uniqueFilePaths.length} files before upsert in workspace ${scanWorkspace}:`,
							deleteError,
						)
						TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
							error: sanitizeErrorMessage(
								deleteError instanceof Error ? deleteError.message : String(deleteError),
							),
							stack:
								deleteError instanceof Error
									? sanitizeErrorMessage(deleteError.stack || "")
									: undefined,
							location: "processRobustBatch:deletePointsByMultipleFilePaths",
							fileCount: uniqueFilePaths.length,
							attemptNumber: attempts,
						})
						throw new Error(
							`Failed to delete points for ${uniqueFilePaths.length} files. Workspace: ${scanWorkspace}. ${deleteError instanceof Error ? deleteError.message : String(deleteError)}`,
							{ cause: deleteError },
						)
					}
				}

				// --- Embedding Creation Step ---
				console.log(`Creating embeddings for batch of ${batchTexts.length} texts (attempt ${attempts})`)

				const startTime = Date.now()
				const { embeddings } = await this.embedder.createEmbeddings(batchTexts)
				const embeddingTime = Date.now() - startTime

				console.log(`Embeddings created in ${embeddingTime}ms`)

				// --- Vector Store Upsert Step ---
				const points = batchBlocks.map((block, index) => {
					const normalizedAbsolutePath = generateNormalizedAbsolutePath(block.file_path, scanWorkspace)
					const pointId = uuidv5(block.segmentHash, QDRANT_CODE_BLOCK_NAMESPACE)

					return {
						id: pointId,
						vector: embeddings[index],
						payload: {
							filePath: generateRelativeFilePath(normalizedAbsolutePath, scanWorkspace),
							codeChunk: block.content,
							startLine: block.start_line,
							endLine: block.end_line,
							segmentHash: block.segmentHash,
						},
					}
				})

				console.log(`Upserting ${points.length} points to vector store`)
				const upsertStartTime = Date.now()
				await this.qdrantClient.upsertPoints(points)
				const upsertTime = Date.now() - upsertStartTime

				console.log(`Points upserted in ${upsertTime}ms`)
				onBlocksIndexed?.(batchBlocks.length)

				// --- Cache Update Step ---
				for (const fileInfo of batchFileInfos) {
					await this.cacheManager.updateHash(fileInfo.filePath, fileInfo.fileHash)
				}

				success = true
				console.log(`Batch processed successfully in ${Date.now() - startTime}ms (attempt ${attempts})`)
			} catch (error) {
				lastError = error as Error
				console.error(
					`[DirectoryScanner] Error processing batch (attempt ${attempts}/${MAX_BATCH_RETRIES}) in workspace ${scanWorkspace}:`,
					error,
				)

				TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
					error: sanitizeErrorMessage(error instanceof Error ? error.message : String(error)),
					stack: error instanceof Error ? sanitizeErrorMessage(error.stack || "") : undefined,
					location: "processRobustBatch:retry",
					attemptNumber: attempts,
					batchSize: batchBlocks.length,
					workspace: scanWorkspace,
				})

				// Check if we should split the batch on certain errors
				const shouldSplitBatch =
					error instanceof Error &&
					(error.message.includes("timeout") ||
						error.message.includes("too large") ||
						error.message.includes("memory") ||
						batchBlocks.length > ADAPTIVE_BATCH_SIZE_MIN * 2)

				if (shouldSplitBatch && attempts === 1 && batchBlocks.length > ADAPTIVE_BATCH_SIZE_MIN) {
					console.log(
						`Splitting large batch into smaller chunks due to error: ${error instanceof Error ? error.message : String(error)}`,
					)

					// Split batch in half and process recursively
					const midPoint = Math.floor(batchBlocks.length / 2)

					const firstHalf = {
						blocks: batchBlocks.slice(0, midPoint),
						texts: batchTexts.slice(0, midPoint),
						fileInfos: batchFileInfos.slice(0, midPoint),
					}

					const secondHalf = {
						blocks: batchBlocks.slice(midPoint),
						texts: batchTexts.slice(midPoint),
						fileInfos: batchFileInfos.slice(midPoint),
					}

					try {
						await this.processRobustBatch(
							firstHalf.blocks,
							firstHalf.texts,
							firstHalf.fileInfos,
							scanWorkspace,
							onError,
							onBlocksIndexed,
						)

						await this.processRobustBatch(
							secondHalf.blocks,
							secondHalf.texts,
							secondHalf.fileInfos,
							scanWorkspace,
							onError,
							onBlocksIndexed,
						)

						success = true
						console.log(`Batch successfully processed after splitting`)
						break
					} catch (splitError) {
						console.error(`Failed to process split batches:`, splitError)
						lastError = splitError as Error
						// Continue to normal retry logic
					}
				}

				// Normal retry logic with exponential backoff
				if (attempts < MAX_BATCH_RETRIES && !success) {
					const baseDelay = INITIAL_RETRY_DELAY_MS * Math.pow(2, attempts - 1)
					const jitter = Math.random() * 1000 // Add jitter to prevent thundering herd
					const delay = Math.min(baseDelay + jitter, EXPONENTIAL_BACKOFF_MAX_MS)

					console.warn(
						`Retrying batch in ${Math.round(delay)}ms (attempt ${attempts + 1}/${MAX_BATCH_RETRIES})`,
					)
					await new Promise((resolve) => setTimeout(resolve, delay))
				}
			}
		}

		if (!success && lastError) {
			console.error(
				`[DirectoryScanner] Failed to process batch after ${MAX_BATCH_RETRIES} attempts and batch splitting`,
			)

			// Capture final failure telemetry
			TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
				error: sanitizeErrorMessage(lastError.message),
				stack: sanitizeErrorMessage(lastError.stack || ""),
				location: "processRobustBatch:finalFailure",
				maxRetries: MAX_BATCH_RETRIES,
				batchSize: batchBlocks.length,
				workspace: scanWorkspace,
			})

			if (onError) {
				// Preserve the original error message from embedders which now have detailed i18n messages
				const errorMessage = lastError.message || "Unknown error"

				onError(
					new Error(
						t("embeddings:scanner.failedToProcessBatchWithError", {
							maxRetries: MAX_BATCH_RETRIES,
							errorMessage,
						}) + ` (Workspace: ${scanWorkspace}, Batch Size: ${batchBlocks.length})`,
					),
				)
			}
		}
	}
}
