import my_sqlite as msq 
import sys
import subprocess
from easydict import EasyDict as edict

def prms1():
	p = edict()
	p.a = None
	p.b = 1
	return p

def prms2():
	p = prms1()
	p.c = None
	p.d = 'check'
	return p

def prms3():
	p = prms2()
	p.e = 34.56
	p.f = None
	return p

def test1():
	dbFile = 'test_data/test-sql.sqlite'
	db     = msq.SqDb(dbFile)
	p1, p2, p3 = prms1(), prms2(), prms3()
	db.fetch(p3)
	idx = db.get_id(p1)							
	assert idx is None
	idx = db.get_id(p2)
	assert idx is None
	p3Idx = db.get_id(p3)
	assert p3Idx is not None
	db.fetch(p1)
	idx = db.get_id(p3)
	assert p3Idx==idx
	idx = db.get_id(p2)
	assert idx is None
	idx  = db.get_id(p1)
	assert idx is not None	
	subprocess.check_call(['rm %s' % dbFile],shell=True)	

def test2():
	dbFile = 'test_data/test-sql.sqlite'
	db     = msq.SqDb(dbFile)
	p1, p2, p3 = prms1(), prms2(), prms3()
	db.fetch(p2)
	idx = db.get_id(p1)							
	assert idx is None
	p2Idx = db.get_id(p2)
	assert p2Idx is not None
	p3Idx = db.get_id(p3)
	assert p3Idx is None
	db.close()
	del db
	db = msq.SqDb(dbFile)
	db.fetch(p2)
	db.fetch(p1)
	idx = db.get_id(p2)
	assert idx == p2Idx	
	idx = db.get_id(p3)
	assert idx is None	
	idx = db.get_id(p1)
	assert idx is not None	
	db.fetch(p1)
	print (db._get({}))
	subprocess.check_call(['rm %s' % dbFile],shell=True)	
