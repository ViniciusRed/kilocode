import { useState, useEffect } from "react"
import { vscode } from "@/utils/vscode"
import { useExtensionState } from "@/context/ExtensionStateContext"
import { useAppTranslation } from "@/i18n/TranslationContext"
import { VSCodeDropdown, VSCodeOption } from "@vscode/webview-ui-toolkit/react"
import { ProfileDataResponsePayload, WebviewMessage, UserOrganizationWithApiKey } from "@roo/WebviewMessage"

export const OrganizationSelector = () => {
	const [organizations, setOrganizations] = useState<UserOrganizationWithApiKey[]>([])
	const { apiConfiguration, currentApiConfigName } = useExtensionState()
	const [selectedOrgId, setSelectedOrgId] = useState<string>(apiConfiguration?.kilocodeOrganizationId ?? "personal")
	const { t } = useAppTranslation()

	useEffect(() => {
		if (!apiConfiguration?.kilocodeToken) return

		vscode.postMessage({
			type: "fetchProfileDataRequest",
		})
	}, [apiConfiguration?.kilocodeToken])

	useEffect(() => {
		const handleMessage = (event: MessageEvent<WebviewMessage>) => {
			const message = event.data
			if (message.type === "profileDataResponse") {
				const payload = message.payload as ProfileDataResponsePayload
				if (payload.success) {
					setOrganizations(payload.data?.organizations ?? [])
				} else {
					console.error("Error fetching profile organizations data:", payload.error)
					setOrganizations([])
				}
			} else if (message.type === "updateProfileData") {
				vscode.postMessage({
					type: "fetchProfileDataRequest",
				})
			}
		}

		window.addEventListener("message", handleMessage)
		return () => {
			window.removeEventListener("message", handleMessage)
		}
	}, [apiConfiguration?.kilocodeToken])

	useEffect(() => {
		if (selectedOrgId === "personal") {
			// Switch back to personal account - clear organization token
			vscode.postMessage({
				type: "upsertApiConfiguration",
				text: currentApiConfigName,
				apiConfiguration: {
					...apiConfiguration,
					kilocodeOrganizationId: undefined,
				},
			})
			vscode.postMessage({
				type: "fetchBalanceDataRequest",
				values: {
					apiKey: apiConfiguration?.kilocodeToken,
				},
			})
		} else {
			const org = organizations.find((o) => o.id === selectedOrgId)
			if (org) {
				vscode.postMessage({
					type: "upsertApiConfiguration",
					text: currentApiConfigName,
					apiConfiguration: {
						...apiConfiguration,
						kilocodeOrganizationId: org.id,
					},
				})
				vscode.postMessage({
					type: "fetchBalanceDataRequest",
					values: {
						apiKey: org.apiKey,
					},
				})
			}
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [organizations, selectedOrgId])

	const handleOrgChange = (e: any) => {
		const newOrgId = e.target.value
		setSelectedOrgId(newOrgId)
	}

	if (!organizations.length) return null

	return (
		<div className="mb-6">
			<label className="text-sm text-[var(--vscode-descriptionForeground)] block mb-2">
				{t("kilocode:profile.organization")}
			</label>
			<VSCodeDropdown value={selectedOrgId} onChange={handleOrgChange} className="w-full">
				<VSCodeOption value="personal">{t("kilocode:profile.personal")}</VSCodeOption>
				{organizations.map((org) => (
					<VSCodeOption key={org.id} value={org.id}>
						{org.name} ({org.role.toUpperCase()})
					</VSCodeOption>
				))}
			</VSCodeDropdown>
		</div>
	)
}
