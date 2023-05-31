"use client";

import { GoogleDriveIcon } from "@/components/icons/icons";
import useSWR, { useSWRConfig } from "swr";
import { fetcher } from "@/lib/fetcher";
import { LoadingAnimation } from "@/components/Loading";
import { useRouter } from "next/navigation";
import { Popup, PopupSpec } from "@/components/admin/connectors/Popup";
import { useState } from "react";
import { HealthCheckBanner } from "@/components/health/healthcheck";
import { Button } from "@/components/Button";
import {
  Connector,
  ConnectorBase,
  ConnectorIndexingStatus,
  Credential,
  GoogleDriveCredentialJson,
} from "@/lib/types";
import { deleteConnector } from "@/lib/connector";
import { StatusRow } from "@/components/admin/connectors/table/ConnectorsTable";

const AppCredentialUpload = ({
  setPopup,
}: {
  setPopup: (popupSpec: PopupSpec | null) => void;
}) => {
  const [appCredentialJsonStr, setAppCredentialJsonStr] = useState<
    string | undefined
  >();

  return (
    <>
      <input
        className={
          "mr-3 text-sm text-gray-900 border border-gray-300 rounded-lg " +
          "cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none " +
          "dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400"
        }
        type="file"
        accept=".json"
        onChange={(event) => {
          if (!event.target.files) {
            return;
          }
          const file = event.target.files[0];
          const reader = new FileReader();

          reader.onload = function (loadEvent) {
            if (!loadEvent?.target?.result) {
              return;
            }
            const fileContents = loadEvent.target.result;
            setAppCredentialJsonStr(fileContents as string);
          };

          reader.readAsText(file);
        }}
      />

      <Button
        disabled={!appCredentialJsonStr}
        onClick={async () => {
          const response = await fetch(
            "/api/admin/connector/google-drive/app-credential",
            {
              method: "PUT",
              headers: {
                "Content-Type": "application/json",
              },
              body: appCredentialJsonStr,
            }
          );
          if (response.ok) {
            setPopup({
              message: "Successfully uploaded app credentials",
              type: "success",
            });
          } else {
            setPopup({
              message: `Failed to upload app credentials - ${response.status}`,
              type: "error",
            });
          }
        }}
      >
        Upload
      </Button>
    </>
  );
};

const Main = () => {
  const router = useRouter();
  const { mutate } = useSWRConfig();

  const {
    data: appCredentialData,
    isLoading: isAppCredentialLoading,
    error: isAppCredentialError,
  } = useSWR<{ client_id: string }>(
    "/api/admin/connector/google-drive/app-credential",
    fetcher
  );
  const {
    data: connectorIndexingStatuses,
    isLoading: isConnectorIndexingStatusesLoading,
    error: isConnectorIndexingStatusesError,
  } = useSWR<ConnectorIndexingStatus<any>[]>(
    "/api/admin/connector/indexing-status",
    fetcher
  );
  const {
    data: credentialsData,
    isLoading: isCredentialsLoading,
    error: isCredentialsError,
  } = useSWR<Credential<GoogleDriveCredentialJson>[]>(
    "/api/admin/credential",
    fetcher
  );

  const [popup, setPopup] = useState<{
    message: string;
    type: "success" | "error";
  } | null>(null);
  const setPopupWithExpiration = (popupSpec: PopupSpec | null) => {
    setPopup(popupSpec);
    setTimeout(() => {
      setPopup(null);
    }, 4000);
  };

  if (
    isCredentialsLoading ||
    isAppCredentialLoading ||
    isConnectorIndexingStatusesLoading
  ) {
    return (
      <div className="mx-auto">
        <LoadingAnimation text="" />
      </div>
    );
  }

  if (isCredentialsError || !credentialsData) {
    return (
      <div className="mx-auto">
        <div className="text-red-500">Failed to load credentials.</div>
      </div>
    );
  }

  if (isConnectorIndexingStatusesError || !connectorIndexingStatuses) {
    return (
      <div className="mx-auto">
        <div className="text-red-500">Failed to load connectors.</div>
      </div>
    );
  }

  if (isAppCredentialError) {
    return (
      <div className="mx-auto">
        <div className="text-red-500">
          Error loading Google Drive app credentials. Contact an administrator.
        </div>
      </div>
    );
  }

  const googleDriveConnectorIndexingStatuses: ConnectorIndexingStatus<{}>[] =
    connectorIndexingStatuses.filter(
      (connectorIndexingStatus) =>
        connectorIndexingStatus.connector.source === "google_drive"
    );
  const googleDriveConnectorIndexingStatus =
    googleDriveConnectorIndexingStatuses[0];
  const googleDriveCredential = credentialsData.filter(
    (credential) => credential.credential_json?.google_drive_tokens
  )[0];

  return (
    <>
      {popup && <Popup message={popup.message} type={popup.type} />}
      <h2 className="font-bold mb-2 mt-6 ml-auto mr-auto">
        Step 1: Provide your app Credentials
      </h2>
      <div className="mt-2">
        {appCredentialData?.client_id ? (
          <div className="text-sm">
            <div>
              Found existing app credentials with the following{" "}
              <b>Client ID:</b>
              <p className="italic mt-1">{appCredentialData.client_id}</p>
            </div>
            <div className="mt-4">
              If you want to update these credentials, upload a new
              credentials.json file below.
              <div className="mt-2">
                <AppCredentialUpload
                  setPopup={(popup) => {
                    mutate("/api/admin/connector/google-drive/app-credential");
                    setPopupWithExpiration(popup);
                  }}
                />
              </div>
            </div>
          </div>
        ) : (
          <>
            <p className="text-sm">
              Follow the guide{" "}
              <a
                className="text-blue-500"
                target="_blank"
                href="https://docs.danswer.dev/connectors/google_drive#authorization"
              >
                here
              </a>{" "}
              to setup your google app in your company workspace. Download the
              credentials.json, and upload it here.
            </p>
            <AppCredentialUpload
              setPopup={(popup) => {
                mutate("/api/admin/connector/google-drive/app-credential");
                setPopupWithExpiration(popup);
              }}
            />
          </>
        )}
      </div>

      <h2 className="font-bold mb-2 mt-6 ml-auto mr-auto">
        Step 2: Authenticate with Danswer
      </h2>
      <div className="text-sm mb-4">
        {googleDriveCredential ? (
          <p>
            <i>Existing credential already setup!</i> If you want to reset that
            credential, click the button below to go through the OAuth flow
            again.
          </p>
        ) : (
          <>
            <p>
              Next, you must provide credentials via OAuth. This gives us read
              access to the docs you have access to in your google drive
              account.
            </p>
          </>
        )}
      </div>
      <Button
        onClick={async () => {
          const credentialCreationResponse = await fetch(
            "/api/admin/credential",
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                public_doc: true,
                credential_json: {},
              }),
            }
          );
          if (!credentialCreationResponse.ok) {
            setPopupWithExpiration({
              message: `Failed to create credential - ${credentialCreationResponse.status}`,
              type: "error",
            });
            return;
          }
          const credential =
            (await credentialCreationResponse.json()) as Credential<{}>;

          const authorizationUrlResponse = await fetch(
            `/api/admin/connector/google-drive/authorize/${credential.id}`
          );
          if (!authorizationUrlResponse.ok) {
            setPopupWithExpiration({
              message: `Failed to create credential - ${authorizationUrlResponse.status}`,
              type: "error",
            });
            return;
          }
          const authorizationUrlJson =
            (await authorizationUrlResponse.json()) as { auth_url: string };

          router.push(authorizationUrlJson.auth_url);
        }}
      >
        Authenticate with Google Drive
      </Button>

      <h2 className="font-bold mb-2 mt-6 ml-auto mr-auto">
        Step 3: Start Indexing!
      </h2>
      {googleDriveConnectorIndexingStatus ? (
        <div>
          <div className="text-sm mb-2">
            <div className="flex mb-1">
              The Google Drive connector is setup!{" "}
              <b className="mx-2">Status:</b>{" "}
              <StatusRow
                connectorIndexingStatus={googleDriveConnectorIndexingStatus}
                hasCredentialsIssue={
                  googleDriveConnectorIndexingStatus.connector.credential_ids
                    .length === 0
                }
                setPopup={setPopupWithExpiration}
                onUpdate={() => {
                  mutate("/api/admin/connector/indexing-status");
                }}
              />
            </div>
            <p>
              Checkout the{" "}
              <a href="/admin/indexing/status" className="text-blue-500">
                status page
              </a>{" "}
              for the latest indexing status. We fetch the latest documents from
              Google Drive every <b>10</b> minutes.
            </p>
          </div>
          <Button
            onClick={() => {
              deleteConnector(
                googleDriveConnectorIndexingStatus.connector.id
              ).then(() => {
                setPopupWithExpiration({
                  message: "Successfully deleted connector!",
                  type: "success",
                });
                mutate("/api/admin/connector/indexing-status");
              });
            }}
          >
            Delete Connector
          </Button>
        </div>
      ) : (
        <>
          <p className="text-sm mb-2">
            Click the button below to create a connector. We will refresh the
            latest documents from Google Drive every <b>10</b> minutes.
          </p>
          <Button
            onClick={async () => {
              const connectorBase: ConnectorBase<{}> = {
                name: "GoogleDriveConnector",
                input_type: "load_state",
                source: "google_drive",
                connector_specific_config: {},
                refresh_freq: 60 * 10, // 10 minutes
                disabled: false,
              };
              const connectorCreationResponse = await fetch(
                `/api/admin/connector`,
                {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                  },
                  body: JSON.stringify(connectorBase),
                }
              );
              if (!connectorCreationResponse.ok) {
                setPopupWithExpiration({
                  message: `Failed to create connector - ${connectorCreationResponse.status}`,
                  type: "error",
                });
                return;
              }
              const connector =
                (await connectorCreationResponse.json()) as Connector<{}>;

              const credentialLinkResponse = await fetch(
                `/api/admin/connector/${connector.id}/credential/${googleDriveCredential.id}`,
                {
                  method: "PUT",
                  headers: {
                    "Content-Type": "application/json",
                  },
                }
              );
              if (!credentialLinkResponse.ok) {
                setPopupWithExpiration({
                  message: `Failed to link connector to credential - ${credentialLinkResponse.status}`,
                  type: "error",
                });
                return;
              }

              setPopupWithExpiration({
                message: "Successfully created connector!",
                type: "success",
              });
              mutate("/api/admin/connector/indexing-status");
            }}
          >
            Add
          </Button>
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
        <GoogleDriveIcon size="32" />
        <h1 className="text-3xl font-bold pl-2">Google Drive</h1>
      </div>

      <Main />
    </div>
  );
}
