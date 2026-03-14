import os

class DocumentLoader:
    """
    Document loader to load text documents from a specified folder.
    Assumes documents are in .txt format.
    """
    def __init__(self, folder_path):
        """
        Initialize the document loader.
        :param folder_path: Path to the folder containing documents.
        """
        self.folder_path = folder_path

    def load_documents(self):
        """
        Load all .txt documents from the folder.
        :return: List of document contents as strings.
        """
        documents = []
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(self.folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
        return documents