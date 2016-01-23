## @package my_sqlite
#  Wrapper around sqlite3 to ease reading and writing
#  in pythonic form.

import sqlite3
import uuid

class SqDb(object):
	#Make a Db
	def __init__(self, name):
		#self.c_ = connection
		#Names of columns, the entries contain types
		self.colNames_ = co.OrderedDict()
		self.create_table(tableName='default')

	#Create table
	def create_table(self, vals={}, tableName='default'):
		'''
			vals: a dict containing whose keys are columns in the table
		'''
		assert tableName not in self.colNames_.keys(),'Table %s exists' % tableName
		self.colNames_[tableName] = co.OrderedDict()
		self.colNames_[tableName]['id'] = str
		#Insert the columns
		pass

	#fetch the entry if it exists or create a new one
	def fetch(self, vals={}, tableName='default'):
		if not set(vals.keys()).issubset(set(self.get_column_names())):
			self.add(vals, tableName)
		

	#Add a column to the table
	def _add_column(self, colName, colType, tableName='default'):	
		cmd = 'ALTER TABLE %s ADD COLUMN "%s" "%s"' % (tableName, colName, colType)
		self.c_.execute(cmd)
		self.c_.commit()	

	#Get columns names
	def get_column_names(self, tableName='default'):
		return self.colNames_[tableName].keys()
	
	#Get the names of columns from the db
	def _get_column_names_from_db(self, tableName='default'):
		cs    = self.c_.execute('SELECT * FROM %s' % tableName) 
		names = [description[0] for description in cs.description]	
		return names
		
