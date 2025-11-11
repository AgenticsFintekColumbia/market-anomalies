from ._wrds_base import WRDSDataIngestor
import pandas as pd


class CRSPIngestor(WRDSDataIngestor):
    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_daily(self, start: str, end: str) -> pd.DataFrame:
        q = f"""
        SELECT a.permno, a.date, b.ticker, b.ncusip AS cusip,
               a.prc AS price, a.vol AS volume, a.ret AS return, a.shrout,
               b.exchcd AS exchange_code
        FROM crsp.dsf a
        LEFT JOIN crsp.dsenames b
          ON a.permno=b.permno AND b.namedt<=a.date AND a.date<=b.nameendt
        WHERE a.date BETWEEN '{start}' AND '{end}' AND b.exchcd IN (1,2,3)
        ORDER BY a.permno, a.date
        """
        return self.conn.raw_sql(q)

    @staticmethod
    def schema_doc():
        return {"dataset":"CRSP Daily Stock File","library":"crsp","primary_table":"dsf",
                "date_field":"date","identifier_fields":["permno","ticker","cusip"]}