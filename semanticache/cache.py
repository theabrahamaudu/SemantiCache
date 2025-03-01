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


class SemantiCache:
    def __init__(
            self,
            trim_by_size: bool | None = None,
            cache_path: str = "./sem_cache",
            config_path: str = "./sem_config",
            cache_size: int | None = None,
            ttl: int | None = None,
            threshold: float | None = None,
    ):
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

        self.trim_by_size = trim_by_size
        self.cache_size = cache_size
        self.ttl = ttl
        self.threshold = threshold
        self.server_config = self.__load_config()["cache"]
        self.__check_params()
        self.__update_config_file()

        try:
            self.cache_index = self.__load_cache_index()
            print("loaded cache index from memory during init")
        except Exception as e:
            print(f"unable to load cache index during init: {e}")
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

    def clear(self) -> None:
        """
        Remove all files in the cache directory
        """
        try:
            files = glob.glob(self.cache_path+"/*")
            for f in files:
                os.remove(f)
            self.cache_index = None
        except Exception as e:
            print(f"unable to clear cache: {e}")

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
            else:
                print("trimming by ttl::")
                # delete the oldest records
                curr_time = datetime.now()
                for doc in documents:
                    delta = (
                        curr_time - doc.metadata["updated_at"]
                    ).total_seconds()
                    if delta > self.ttl:
                        self.__delete_record(id=doc.metadata["id"])

        except Exception as e:
            print(f"unable to trim cache: {e}")

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
                    print("loaded cache index from memory during creation")
                except Exception as e:
                    print(f"unable to load cache index during creation: {e}")
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
            print(f"Error creating record: {e}")

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
            print(f"Error reading record: {e}")

    def __update_record(self, record: Document, updated_at: datetime) -> None:
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

    def __delete_record(self, id: str) -> None:
        if self.cache_index:
            _, _ = self.remove(self.cache_index, [id])
            self.__persist_cache_state()

    @staticmethod
    def remove(vectorstore: FAISS, docstore_ids: list[str]):
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
            vectorstore.docstore = {}
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
        n_total = vectorstore.index.ntotal
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
            leaderboard = []

        if not updated:
            leaderboard.append({
                "query": query,
                "response": response,
                "hits": hits
            })

        with open(self.leaderboard_path, 'w', encoding="utf-8") as file:
            json.dump(leaderboard, file, indent=4)

    def read_leaderboard(self, top_n: int = 5) -> list | None:
        try:
            with open(self.leaderboard_path, 'r', encoding="utf-8") as file:
                leaderboard: list = json.load(file)

            if isinstance(leaderboard, list):
                leaderboard = sorted(
                    leaderboard,
                    key=lambda x: x["hits"],
                    reverse=True
                )
                return leaderboard[:top_n]
        except Exception as e:
            print(f"error reading leaderboard {e}")
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
            print("configured new cache index")
            return cache_index
        except Exception as e:
            print(f"error configuring index: {e}")

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
            "threshold": self.def_threshold
        }

        for key, value in defaults.items():
            if getattr(self, key) is None:
                if self.server_config[key] == "None":
                    setattr(self, key, value)
                else:
                    setattr(self, key, self.server_config[key])

    def __load_config(self) -> Dict[str, Any]:
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                config: dict = yaml.safe_load(f)

            return config
        except FileNotFoundError:
            self.__create_directories()
            self.__create_config_file()
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                config: dict = yaml.safe_load(f)

                return config

    def __create_directories(self):
        """Create the cache and config directories if they don't exist."""
        cache_dir = Path(self.cache_path)
        config_dir = Path(self.config_path)

        cache_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)

    def __create_config_file(self):
        """Create the config.yaml file with predefined content."""
        config_path = Path(self.yaml_path)

        if not config_path.exists():
            config_content = f"""cache:
    path: {self.cache_path}
    name: {self.cache_name}
    cache_size: {self.cache_size}
    ttl: {self.ttl}
    threshold: {self.threshold}
    trim_by_size: {self.trim_by_size}
    """
            config_path.write_text(config_content, encoding="utf-8")

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
    """
        config_path.write_text(config_content, encoding="utf-8")
