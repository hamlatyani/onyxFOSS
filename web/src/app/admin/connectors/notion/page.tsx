"use client";

import * as Yup from "yup";
import { NotionIcon, TrashIcon } from "@/components/icons/icons";
import { TextFormField } from "@/components/admin/connectors/Field";
import { HealthCheckBanner } from "@/components/health/healthcheck";
import { CredentialForm } from "@/components/admin/connectors/CredentialForm";
import {
  NotionCredentialJson,
  NotionConfig,
  Credential,
  ConnectorIndexingStatus,
} from "@/lib/types";
import useSWR, { useSWRConfig } from "swr";
import { fetcher } from "@/lib/fetcher";
import { LoadingAnimation } from "@/components/Loading";
import { deleteCredential, linkCredential } from "@/lib/credential";
import { ConnectorForm } from "@/components/admin/connectors/ConnectorForm";
import { ConnectorsTable } from "@/components/admin/connectors/table/ConnectorsTable";
import { usePopup } from "@/components/admin/connectors/Popup";

const Main = () => {
  const { popup, setPopup } = usePopup();

  const { mutate } = useSWRConfig();
  const {
    data: connectorIndexingStatuses,
    isLoading: isConnectorIndexingStatusesLoading,
    error: isConnectorIndexingStatusesError,
  } = useSWR<ConnectorIndexingStatus<any>[]>(
    "/api/manage/admin/connector/indexing-status",
    fetcher
  );
  const {
    data: credentialsData,
    isLoading: isCredentialsLoading,
    error: isCredentialsError,
  } = useSWR<Credential<NotionCredentialJson>[]>(
    "/api/manage/credential",
    fetcher
  );

  if (
    (!connectorIndexingStatuses && isConnectorIndexingStatusesLoading) ||
    (!credentialsData && isCredentialsLoading)
  ) {
    return <LoadingAnimation text="Loading" />;
  }

  if (isConnectorIndexingStatusesError || !connectorIndexingStatuses) {
    return <div>Failed to load connectors</div>;
  }

  if (isCredentialsError || !credentialsData) {
    return <div>Failed to load credentials</div>;
  }

  const notionConnectorIndexingStatuses = connectorIndexingStatuses.filter(
    (connectorIndexingStatus) =>
      connectorIndexingStatus.connector.source === "notion"
  );
  const notionCredential = credentialsData.filter(
    (credential) => credential.credential_json?.notion_integration_token
  )[0];

  return (
    <>
      {popup}
      <h2 className="font-bold mb-2 mt-6 ml-auto mr-auto">
        Step 1: Provide your authorization details
      </h2>

      {notionCredential ? (
        <>
          <div className="flex mb-1 text-sm">
            <p className="my-auto">Existing Integration Token: </p>
            <p className="ml-1 italic my-auto max-w-md">
              {notionCredential.credential_json?.notion_integration_token}
            </p>
            <button
              className="ml-1 hover:bg-gray-700 rounded-full p-1"
              onClick={async () => {
                if (notionConnectorIndexingStatuses.length > 0) {
                  setPopup({
                    type: "error",
                    message:
                      "Must delete all connectors before deleting credentials",
                  });
                  return;
                }
                await deleteCredential(notionCredential.id);
                mutate("/api/manage/credential");
              }}
            >
              <TrashIcon />
            </button>
          </div>
        </>
      ) : (
        <>
          <p className="text-sm">
            To get started you&apos;ll need to create an internal integration in
            Notion for Danswer. Follow the instructions in the&nbsp;
            <a
              href="https://developers.notion.com/docs/create-a-notion-integration"
              target="_blank"
            >
              Notion Developer Documentation
            </a>
            &nbsp; on the Notion website, to create a new integration. Once
            you&apos;ve created an integration, copy the integration secret
            token and paste it below. Follow the remaining instructions on the
            Notion docs to allow Danswer to read Notion Databases and Pages
            using the new integration.
          </p>
          <div className="border-solid border-gray-600 border rounded-md p-6 mt-2 mb-4">
            <CredentialForm<NotionCredentialJson>
              formBody={
                <TextFormField
                  name="notion_integration_token"
                  label="Integration Token:"
                  type="password"
                />
              }
              validationSchema={Yup.object().shape({
                notion_integration_token: Yup.string().required(
                  "Please enter the Notion Integration token for the Danswer integration."
                ),
              })}
              initialValues={{
                notion_integration_token: "",
              }}
              onSubmit={(isSuccess) => {
                if (isSuccess) {
                  mutate("/api/manage/credential");
                  mutate("/api/manage/admin/connector/indexing-status");
                }
              }}
            />
          </div>
        </>
      )}

      {notionConnectorIndexingStatuses.length > 0 && (
        <>
          <h2 className="font-bold mb-2 mt-6 ml-auto mr-auto">
            Notion indexing status
          </h2>
          <p className="text-sm mb-2">
            The latest page updates are fetched from Notion every 10 minutes.
          </p>
          <div className="mb-2">
            <ConnectorsTable<NotionConfig, NotionCredentialJson>
              connectorIndexingStatuses={notionConnectorIndexingStatuses}
              liveCredential={notionCredential}
              getCredential={(credential) => {
                return (
                  <div>
                    <p>{credential.credential_json.notion_integration_token}</p>
                  </div>
                );
              }}
              onCredentialLink={async (connectorId) => {
                if (notionCredential) {
                  await linkCredential(connectorId, notionCredential.id);
                  mutate("/api/manage/admin/connector/indexing-status");
                }
              }}
              onUpdate={() =>
                mutate("/api/manage/admin/connector/indexing-status")
              }
            />
          </div>
        </>
      )}

      {notionCredential && notionConnectorIndexingStatuses.length === 0 && (
        <>
          <div className="border-solid border-gray-600 border rounded-md p-6 mt-4">
            <h2 className="font-bold mb-3">Create Connection</h2>
            <p className="text-sm mb-4">
              Press connect below to start the connection to Notion.
            </p>
            <ConnectorForm<NotionConfig>
              nameBuilder={(values) => `NotionConnector`}
              source="notion"
              inputType="poll"
              formBody={<></>}
              validationSchema={Yup.object().shape({})}
              initialValues={{}}
              refreshFreq={10 * 60} // 10 minutes
              onSubmit={async (isSuccess, responseJson) => {
                if (isSuccess && responseJson) {
                  await linkCredential(responseJson.id, notionCredential.id);
                  mutate("/api/manage/admin/connector/indexing-status");
                }
              }}
            />
          </div>
        </>
      )}

      {!notionCredential && (
        <>
          <p className="text-sm mb-4">
            Please provide your integration details in Step 1 first! Once done
            with that, you&apos;ll be able to start the connection then see
            indexing status.
          </p>
        </>
      )}
    </>
  );
};

export default function Page() {
  return (
    <div className="mx-auto container">
      <div className="mb-4">
        <HealthCheckBanner />
      </div>
      <div className="border-solid border-gray-600 border-b mb-4 pb-2 flex">
        <NotionIcon size="32" />
        <h1 className="text-3xl font-bold pl-2">Notion</h1>
      </div>
      <Main />
    </div>
  );
}
