import os
import glob
from datetime import datetime
import pickle
import json
from pathlib import Path
import yaml
from typing import Any, Dict
from ulid import ULID
import numpy as np
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from semanticache.utils.logger import logger as log_handler


class SemantiCache:

    def __init__(
            self,
            trim_by_size: bool | None = None,
            cache_path: str = "./sem_cache",
            config_path: str = "./sem_config",
            cache_size: int | None = None,
            ttl: int | None = None,
            threshold: float | None = None,
            leaderboard_top_n: int | None = None,
            log_level: str = "WARNING"
    ):
        self.logger = log_handler(log_level)
        self.cache_name = "sem_cache_index"
        self.cache_path = cache_path
        self.config_path = config_path
        self.yaml_path = f"{self.config_path}/sem_config.yaml"
        self.leaderboard_path = f"{self.cache_path}/leaderboard.json"
        self.cache_index_object = f"{self.cache_path}/sem_cache_index.pkl"

        # define default params
        self.def_trim_by_size = True
        self.def_cache_size = 100
        self.def_ttl = 3600
        self.def_threshold = 0.1
        self.def_leaderboard_top_n = 5

        self.trim_by_size = trim_by_size
        self.cache_size = cache_size
        self.ttl = ttl
        self.threshold = threshold
        self.leaderboard_top_n = leaderboard_top_n
        self.server_config = self.__load_config()["cache"]
        self.__check_params()
        self.__update_config_file()

        try:
            self.cache_index = self.__load_cache_index()
            self.logger.info(
                "loaded cache index from '%s' during init" % self.cache_path
            )
        except Exception as e:
            self.logger.warning(
                "unable to load cache index during init: %s" % e
            )
            self.cache_index = None

    def get(self, query: str) -> str | None:
        record = self.__read_record(query)
        if record is not None:
            self.__update_record(record, datetime.now())
            self.__update_leaderboard(
                record.page_content,
                record.metadata["response"],
                int(record.metadata["hits"] + 1),
            )
            return record.metadata["response"]
        return record

    def set(self, query: str, response: str) -> None:
        self.__create_record(
            created_at=datetime.now(),
            query=query,
            response=response,
        )
        self.__trim_cache()

    def clear(self, clear_files: bool = False) -> None:
        """
        Remove all files in the cache directory
        """
        try:
            n_removed, _ = self.remove(self.cache_index, None)  # type: ignore
            self.logger.info(
                "cleared %s records from cache" % n_removed
            )
            if clear_files:
                files = glob.glob(self.cache_path+"/*")
                for f in files:
                    os.remove(f)
                self.cache_index = None
                self.logger.info(
                    "cleared cache files at '%s'" % self.cache_path
                )
        except Exception as e:
            self.logger.exception(
                "unable to clear cache: %s" % e, exc_info=True
            )

    def __trim_cache(self) -> None:
        try:
            index = pickle.load(open(self.cache_index_object, "rb"))
            memory: InMemoryDocstore = index[0]
            ulid_to_id: dict = index[1]

            ids: list[str] = list(ulid_to_id.values())

            documents: list[Document] = []
            for _id in ids:
                documents.append(memory.search(_id))  # type: ignore

            if self.trim_by_size:
                self.logger.info(
                    "trimming cache by size -- %s max" % self.cache_size
                )
                # delete the least hit records
                if len(documents) > self.cache_size:  # type: ignore
                    ref_list = []
                    for doc in documents:
                        ref_list.append({
                            "id": doc.metadata["id"],
                            "hits": doc.metadata["hits"],
                        })
                    sorted_ref_list = sorted(
                        ref_list,
                        key=lambda x: x["hits"],
                        reverse=True
                    )
                    for doc in sorted_ref_list[self.cache_size:]:
                        self.__delete_record(id=doc["id"])
                    self.logger.info(
                        "trimmed off %s records from cache" % len(
                            sorted_ref_list[self.cache_size:]
                        )
                    )
            else:
                self.logger.info("trimming by ttl -- %s secs" % self.ttl)
                # delete the oldest records
                curr_time = datetime.now()
                count = 0
                for doc in documents:
                    delta = (
                        curr_time - doc.metadata["updated_at"]
                    ).total_seconds()
                    if delta > self.ttl:
                        self.__delete_record(id=doc.metadata["id"])
                        count += 1
                self.logger.info("trimmed off %s record(s) from cache" % count)
        except Exception as e:
            self.logger.exception(
                "unable to trim cache: %s " % e, exc_info=True
            )

    def __create_record(
            self,
            created_at: datetime,
            query: str,
            response: str,
            updated_at: datetime | None = None,
            record_id: str | None = None,
            hits: int | None = None,
    ) -> str | None:
        try:
            if hits is None:
                hits = 0
            if updated_at is None:
                updated_at = created_at
            if record_id is None:
                record_id = str(ULID())
            if self.cache_index is None:
                try:
                    self.cache_index = self.__load_cache_index()
                    self.logger.info(
                        "loaded cache index from '%s' for record creation"
                        % self.cache_path
                    )
                except Exception as e:
                    self.logger.exception(
                        "unable to load cache from %s for record creation: %s"
                        % (self.cache_path, e), exc_info=True
                    )
                    self.cache_index = self.__configure_cache_index(
                        query=query,
                        response=response,
                        new_id=record_id
                    )
                    self.__persist_cache_state()
            else:
                self.cache_index.add_texts(
                    texts=[query],
                    embedding=self.__get_embedder(),
                    metadatas=[{
                            "created_at": created_at,
                            "updated_at": updated_at,
                            "response": response,
                            "hits": hits,
                            "id": record_id
                        }],
                    ids=[record_id]
                )
                self.__persist_cache_state()

            return record_id
        except Exception as e:
            self.logger.error("error creating record: %s" % e)

    def __read_record(self, text: str) -> Document | None:
        try:
            if self.cache_index is None:
                return None
            else:
                record = self.cache_index.similarity_search_with_score(
                    text,
                    k=1,
                )
                if len(record) > 0:
                    score = record[0][1]
                    if score <= self.threshold:  # type: ignore
                        return record[0][0]
                return None
        except Exception as e:
            self.logger.error("error reading record: %s" % e)

    def __update_record(self, record: Document, updated_at: datetime) -> None:
        try:
            updated_hits = int(record.metadata["hits"]) + 1
            self.__delete_record(id=record.metadata["id"])
            self.__create_record(
                query=record.page_content,
                created_at=record.metadata["created_at"],
                updated_at=updated_at,
                response=record.metadata["response"],
                hits=updated_hits,
                record_id=record.metadata["id"]
            )
        except Exception as e:
            self.logger.error(
                "error updating record %s in cache: %s"
                % (record.metadata["id"], e)
            )

    def __delete_record(self, id: str) -> None:
        try:
            if self.cache_index:
                _, _ = self.remove(self.cache_index, [id])
                self.__persist_cache_state()
        except Exception as e:
            self.logger.error(
                "error deleting record '%s' from cache: %s"
                % (id, e)
            )

    @staticmethod
    def remove(
        vectorstore: FAISS, docstore_ids: list[str] | None
    ) -> tuple[int, int]:
        """
        Function to remove documents from the vectorstore.

        Parameters
        ----------
        vectorstore : FAISS
            The vectorstore to remove documents from.
        docstore_ids : Optional[List[str]]
            The list of docstore ids to remove.
            If None, all documents are removed.

        Returns
        -------
        n_removed : int
            The number of documents removed.
        n_total : int
            The total number of documents in the vectorstore.

        Raises
        ------
        ValueError
            If there are duplicate ids in the list of ids to remove.
        """
        if docstore_ids is None:
            vectorstore.docstore = {}  # type: ignore
            vectorstore.index_to_docstore_id = {}
            n_removed = vectorstore.index.ntotal
            n_total = vectorstore.index.ntotal
            vectorstore.index.reset()
            return n_removed, n_total
        set_ids = set(docstore_ids)
        if len(set_ids) != len(docstore_ids):
            raise ValueError("Duplicate ids in list of ids to remove.")
        index_ids = [
            i_id
            for i_id, d_id in vectorstore.index_to_docstore_id.items()
            if d_id in docstore_ids
        ]
        n_removed = len(index_ids)
        n_total: int = vectorstore.index.ntotal
        vectorstore.index.remove_ids(np.array(index_ids, dtype=np.int64))
        for i_id, d_id in zip(index_ids, docstore_ids):
            del vectorstore.docstore._dict[  # type: ignore
                d_id
            ]  # remove the document from the docstore

            del vectorstore.index_to_docstore_id[
                i_id
            ]  # remove the index to docstore id mapping
        vectorstore.index_to_docstore_id = {
            i: d_id
            for i, d_id in enumerate(vectorstore.index_to_docstore_id.values())
        }
        return n_removed, n_total

    def __update_leaderboard(
            self,
            query: str,
            response: str,
            hits: int
    ) -> None:
        record = {
            "query": query,
            "response": response,
            "hits": hits
        }

        updated = False
        try:
            with open(self.leaderboard_path, 'r', encoding="utf-8") as file:
                leaderboard: list = json.load(file)

            for record in leaderboard:
                if record["query"] == query:
                    record["hits"] = hits
                    updated = True
                    break
        except Exception:
            self.logger.warning(
                "no leaderboard file found at %s. creating new file"
                % self.leaderboard_path
            )
            leaderboard = []

        if not updated:
            leaderboard.append({
                "query": query,
                "response": response,
                "hits": hits
            })

        if len(leaderboard) > self.leaderboard_top_n:  # type: ignore
            leaderboard = sorted(
                leaderboard,
                key=lambda x: x["hits"],
                reverse=True
            )
            leaderboard = leaderboard[:self.leaderboard_top_n]

        with open(self.leaderboard_path, 'w', encoding="utf-8") as file:
            json.dump(leaderboard, file, indent=4)
        self.logger.info("updated leaderboard at %s" % self.leaderboard_path)

    def read_leaderboard(self) -> list | None:
        try:
            with open(self.leaderboard_path, 'r', encoding="utf-8") as file:
                leaderboard: list = json.load(file)
            return leaderboard
        except Exception as e:
            self.logger.error("error reading leaderboard %s" % e)
            return None

    @staticmethod
    def __get_embedder(
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            device: str = "cpu",
            normalize_embeddings: bool = False,
    ) -> HuggingFaceEmbeddings:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings}
        )
        return embeddings

    def __configure_cache_index(
            self,
            query: str,
            response: str,
            new_id: str
    ) -> FAISS | None:
        try:
            cache_index = FAISS.from_texts(
                texts=[query],
                embedding=self.__get_embedder(),
                metadatas=[{
                        "created_at": datetime.now(),
                        "updated_at": datetime.now(),
                        "response": response,
                        "hits": 0,
                        "id": new_id,
                    }],
                ids=[new_id]
            )
            self.logger.info(
                "configured cache index at '%s'" % self.cache_path
            )
            return cache_index
        except Exception as e:
            self.logger.error(
                "error configuring cache index at %s: %s"
                % self.cache_path, e
            )

    def __load_cache_index(self) -> FAISS:
        cache_index = FAISS.load_local(
            folder_path=self.cache_path,
            embeddings=self.__get_embedder(),
            index_name=self.cache_name,
            allow_dangerous_deserialization=True
        )
        return cache_index

    def __persist_cache_state(self) -> None:
        if self.cache_index:
            self.cache_index.save_local(
                self.cache_path,
                index_name=self.cache_name
            )

    def __check_params(self) -> None:
        defaults = {
            "trim_by_size": self.def_trim_by_size,
            "cache_size": self.def_cache_size,
            "ttl": self.def_ttl,
            "threshold": self.def_threshold,
            "leaderboard_top_n": self.def_leaderboard_top_n
        }

        for key, value in defaults.items():
            if getattr(self, key) is None:
                if self.server_config[key] == "None":
                    setattr(self, key, value)
                else:
                    setattr(self, key, self.server_config[key])

            if getattr(self, key) != self.server_config[key]:
                self.logger.warning(
                    "overiding %s in %s as '%s' defined in %s instantiation"
                    % (key, self.yaml_path, getattr(self, key), __name__)
                )

    def __load_config(self) -> Dict[str, Any]:
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                config: dict = yaml.safe_load(f)
            self.logger.info(
                "loaded config from %s" % self.yaml_path
            )
            return config
        except FileNotFoundError:
            self.logger.warning(
                "config file not found. attempting to initialize paths"
            )
            self.__create_directories()
            self.__create_config_file()
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                config: dict = yaml.safe_load(f)
                self.logger.info(
                    "loaded config from %s" % self.yaml_path
                )
                return config

    def __create_directories(self):
        """Create the cache and config directories if they don't exist."""
        cache_dir = Path(self.cache_path)
        config_dir = Path(self.config_path)

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "created cache dir at '%s'" % self.cache_path
        )
        config_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "created config dir at '%s'" % self.config_path
        )

    def __create_config_file(self):
        """Create yaml file with predefined content."""
        config_path = Path(self.yaml_path)

        if not config_path.exists():
            config_content = f"""cache:
    path: {self.cache_path}
    name: {self.cache_name}
    cache_size: {self.cache_size}
    ttl: {self.ttl}
    threshold: {self.threshold}
    trim_by_size: {self.trim_by_size}
    leaderboard_top_n: {self.leaderboard_top_n}
    """
            config_path.write_text(config_content, encoding="utf-8")
            self.logger.info(
                "created config file at '%s'" % self.yaml_path
            )

    def __update_config_file(self):
        """Update the config.yaml file with predefined content."""
        config_path = Path(self.yaml_path)

        config_content = f"""cache:
    path: {self.cache_path}
    name: {self.cache_name}
    cache_size: {self.cache_size}
    ttl: {self.ttl}
    threshold: {self.threshold}
    trim_by_size: {self.trim_by_size}
    leaderboard_top_n: {self.leaderboard_top_n}
    """
        config_path.write_text(config_content, encoding="utf-8")
        self.logger.info(
            "updated config file at '%s'" % self.yaml_path
        )
