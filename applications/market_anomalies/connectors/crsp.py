from ._wrds_base import WRDSDataIngestor
import pandas as pd


class CRSPIngestor(WRDSDataIngestor):
    def fetch_if_needed(self, name: str, start: str, end: str) -> pd.DataFrame:
        if self.data_exists(name):
            return self.load_data(name)

        df = self.fetch_daily_stock_data(start, end)
        self.save_data(df, name)
        return df

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_daily(self, start: str, end: str) -> pd.DataFrame:
        q = f"""
            SELECT
                -- core stock data
                a.permno, -- security ID
                a. permco, -- company ID
                a.date, -- time key
                a.prc, -- price
                a.shrout, -- shares outstanding
                a.ret, -- return
                a.exchcd,
                
                -- delisting returns
                b.dlret,
                
                -- compustat link link codes
                c.linktype,
                c.linkdt,
                c.linkenddt,

                -- to get the final corrected return
                COALESCE(b.dlret, a.ret) AS final_ret


            FROM
                -- primary monthly stock data
                crsp.dsf AS a

                -- filter by delisting events
            LEFT JOIN
                crsp.dsedelist AS b
                ON a.permno = b.permno
                AND a.date = b.dlstdt

                -- add accounting data from compustat
            LEFT JOIN
                crsp.ccmxpf_linktable AS c
                ON a.permno = c.lpermno
                AND a.date >= c.linkdt
                AND (a.date <= c.linkenddt OR c.linkenddt IS NULL)

                -- filter major exchanges and by desired dates
            WHERE
                a.exchcd IN (1, 2, 3) -- NYSE, AEX, NASDAQ
                AND a.date BETWEEN '{start}' AND '{end}'
            ORDER BY
                a.permno, a.data
        """

        return self.conn.raw_sql(q)

    def fetch_daily_stock_data(self, start: str, end: str) -> pd.DataFrame:
        return self.fetch_daily(start, end)

    @staticmethod
    def get_schema_documentation():
        return {
            "dataset":"CRSP Daily Stock File",
            "library":"crsp",
            "primary_table":"msf",
            "date_field":"date",
            "identifier_fields":["permno","ticker","cusip"]
        }
