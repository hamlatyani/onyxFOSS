from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# CORRECTION: Importation des classes WebSearchProvider, WebSearchResult et WebContent
from onyx.agents.agent_search.dr.sub_agents.web_search.models import (
    WebSearchProvider,
    WebSearchResult,
    WebContent,
)
from onyx.configs.chat_configs import GOOGLE_API_KEY, GOOGLE_CSE_ID
from onyx.utils.retry_wrapper import retry_builder
from onyx.utils.logger import setup_logger

import requests
from requests.adapters import HTTPAdapter, Retry # Importation pour les sessions
from bs4 import BeautifulSoup
import concurrent.futures # AJOUT: Pour l'exécution parallèle

logger = setup_logger()

# CORRECTION: Hérite de WebSearchProvider
class GoogleClient(WebSearchProvider):
    def __init__(
        self, api_key: str | None = GOOGLE_API_KEY, cse_id: str | None = GOOGLE_CSE_ID
    ) -> None:
        if not api_key or not cse_id:
            raise ValueError(
                "Google API key and CSE ID must be set in environment variables."
            )
        try:
            self.service = build("customsearch", "v1", developerKey=api_key)
            self.cse_id = cse_id
        except HttpError as e:
            logger.error(f"Failed to build Google Custom Search service: {e}")
            raise ValueError("Failed to initialize Google search client.")
        
        # AJOUT: Créer une session requests pour la réutilisation des connexions
        # et la configuration centralisée (User-Agent, timeouts).
        self.scraping_session = requests.Session()
        self.scraping_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
        })
        # AJOUT: Configurer des re-essais (retries) pour le scraping
        # (Ex: 3 essais, avec backoff, pour les codes d'erreur 5xx)
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.scraping_session.mount("https://", HTTPAdapter(max_retries=retries))
        self.scraping_session.mount("http://", HTTPAdapter(max_retries=retries))


    @retry_builder(tries=3, delay=1, backoff=2)
    # CORRECTION: Type de retour WebSearchResult
    def search(self, query: str) -> list[WebSearchResult]:
        try:
            response = (
                self.service.cse()
                .list(q=query, cx=self.cse_id, num=10)
                .execute()
            )
            results = response.get("items", [])
            return [
                # CORRECTION: Utilisation de WebSearchResult
                WebSearchResult(
                    title=result.get("title", ""),
                    link=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    author=None,
                    published_date=None, # Ces champs ne sont généralement pas dans les snippets CSE
                )
                for result in results
            ]
        except HttpError as e:
            logger.error(f"Google Search API call failed: {e}")
            return []

    # AJOUT: Fonction d'aide pour le scraping d'UNE seule URL.
    # C'est cette fonction que nous allons paralléliser.
    def _fetch_and_parse(self, url: str) -> WebContent:
        """
        Scrape le contenu d'une seule URL en utilisant la session partagée.
        """
        try:
            # AJOUT: Utilisation de la session et ajout d'un TIMEOUT
            # (ex: 10 secondes au total pour se connecter et lire la réponse)
            response = self.scraping_session.get(url, timeout=10)
            response.raise_for_status() # Lève une exception pour les codes 4xx/5xx

            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else url
            full_content = ' '.join(soup.stripped_strings)

            # CORRECTION: Utilisation de WebContent
            return WebContent(
                title=title,
                link=url,
                full_content=full_content,
                scrape_successful=True,
                published_date=None,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return WebContent(
                title=url,
                link=url,
                full_content=f"Error: Failed to fetch content. Reason: {e}",
                scrape_successful=False,
                published_date=None,
            )
        except Exception as e:
            # Capture les erreurs de parsing BeautifulSoup ou autres
            logger.error(f"Error parsing content from {url}: {e}")
            return WebContent(
                title=url,
                link=url,
                full_content=f"Error: Failed to parse content. Reason: {e}",
                scrape_successful=False,
                published_date=None,
            )

    # CORRECTION: Type de retour WebContent
    def contents(self, urls: list[str]) -> list[WebContent]:
        if not urls:
            return []

        # AJOUT: Utilisation d'un ThreadPoolExecutor pour exécuter
        # _fetch_and_parse en parallèle pour chaque URL.
        # Nous limitons à 10 workers (threads) max pour ne pas surcharger.
        max_workers = min(len(urls), 10)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # executor.map applique la fonction _fetch_and_parse à chaque url de la liste
            # et récupère les résultats dans l'ordre.
            results = list(executor.map(self._fetch_and_parse, urls))
            
        return results