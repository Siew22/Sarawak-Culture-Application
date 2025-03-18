from databases import Database
test_url = "mssql+pyodbc://terencetu_9899:650771_Tt@poppy.database.windows.net/login_page?driver=ODBC+Driver+18+for+SQL+Server"
try:
    db_test = Database(test_url)
    logger.info("Test URL is valid")
except Exception as e:
    logger.error(f"Test URL failed: {str(e)}")