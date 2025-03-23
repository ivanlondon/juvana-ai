from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_user_documents(user_id: str, domain: str = "immigration") -> list:
    path = f"./data/{user_id}/{domain}/"
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()
