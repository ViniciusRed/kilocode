import { VSCodeRadioGroup, VSCodeRadio, VSCodeTextField } from "@vscode/webview-ui-toolkit/react"
import { ProviderSettings } from "@roo-code/types"
import { useAppTranslation } from "@/i18n/TranslationContext"

export const FastApplySettings = ({
	apiConfiguration,
	setApiConfigurationField,
}: {
	apiConfiguration: ProviderSettings
	setApiConfigurationField: <K extends keyof ProviderSettings>(field: K, value: ProviderSettings[K]) => void
}) => {
	const { t } = useAppTranslation()

	return (
		<div className="space-y-4">
			<VSCodeRadioGroup
				value={(apiConfiguration as any).fastApplyMethod || "morph"}
				onChange={(e) => setApiConfigurationField("fastApplyMethod" as any, (e.target as any)?.value)}>
				<VSCodeRadio value="morph" checked={(apiConfiguration as any).fastApplyMethod === "morph"}>
					{t("settings:experimental.FAST_APPLY.methods.morph")}
				</VSCodeRadio>
				<VSCodeRadio value="ollama" checked={(apiConfiguration as any).fastApplyMethod === "ollama"}>
					{t("settings:experimental.FAST_APPLY.methods.ollama")}
				</VSCodeRadio>
			</VSCodeRadioGroup>

			{(apiConfiguration as any).fastApplyMethod === "morph" && (
				<div className="space-y-3 border-l-2 border-vscode-badgeBackground pl-4">
					<VSCodeTextField
						type="password"
						value={apiConfiguration?.morphApiKey || ""}
						placeholder={t("settings:experimental.FAST_APPLY.morphApiKeyPlaceholder")}
						onChange={(e) => setApiConfigurationField("morphApiKey", (e.target as any)?.value || "")}
						className="w-full">
						{t("settings:experimental.FAST_APPLY.morphApiKey")}
					</VSCodeTextField>
					<div className="text-xs px-1 py-1 flex gap-2 text-vscode-descriptionForeground">
						<span className="codicon codicon-info" />
						<span>{t("settings:experimental.FAST_APPLY.morphWarning")}</span>
					</div>
				</div>
			)}

			{(apiConfiguration as any).fastApplyMethod === "ollama" && (
				<div className="space-y-3 border-l-2 border-vscode-badgeBackground pl-4">
					<VSCodeTextField
						value={apiConfiguration?.ollamaFastApplyModelId || ""}
						placeholder={t("settings:experimental.FAST_APPLY.ollamaModelPlaceholder")}
						onChange={(e) =>
							setApiConfigurationField("ollamaFastApplyModelId", (e.target as any)?.value || "")
						}
						className="w-full">
						{t("settings:experimental.FAST_APPLY.ollamaModelLabel")}
					</VSCodeTextField>
					<div className="text-xs px-1 py-1 flex gap-2 text-vscode-descriptionForeground">
						<span className="codicon codicon-info" />
						<span>{t("settings:experimental.FAST_APPLY.ollamaWarning")}</span>
					</div>
				</div>
			)}
		</div>
	)
}
