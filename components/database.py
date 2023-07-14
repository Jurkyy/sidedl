from components import Component
from packages.database.sqlite.db_save import DBSave
from packages.database.sqlite.db_update import DBUpdate
from packages.database.sqlite.db_select import DBSelect
from packages.database.sqlite.db_delete import DBDelete


class Database(Component):

    def __init__(self, settings, database_root_path):
        super().__init__(settings)
        self.analysis_id = None
        self.database_root_path = database_root_path
        self.set_database_root_path()
        self.db_save = None
        self.db_update = None
        self.db_select = None
        self.db_delete = None

    def set_database_root_path(self):
        self.settings['database_root_path'] = self.database_root_path

    def create(self, database_name):
        self.settings['database_name'] = database_name
        self.db_save = DBSave(f"{self.settings['database_root_path']}{self.settings['database_name']}")
        self.analysis_id = self.db_save.save_analysis(self.settings['database_name'], self.settings["filename"], self.settings)
        self.db_save.analysis_id = self.analysis_id
        self.db_update = DBUpdate(f"{self.settings['database_root_path']}{self.settings['database_name']}", self.analysis_id)
