import pandas as pd
import matplotlib.pyplot as plt

from pymavlink import DFReader
from typing import List, Optional, Dict


class FlightParser:
    def __init__(self,
                 log_path: str) -> None:
        self.log_path: str = log_path
        self.binary_log: str = DFReader.DFReader_binary(
            filename=log_path)
        # Peek what’s available:
        print("Message types present:", sorted(
            [fmt.name for fmt in self.binary_log.formats.values()]))
        print("Counts by type (non-zero):",
              {self.binary_log.id_to_name[i]: c for i, c in enumerate(self.binary_log.counts) if c > 0 and i in self.binary_log.id_to_name})
        # we do this to reset the log for future reads
        self.binary_log.rewind()

    def add_rel_time(self, df: pd.DataFrame, col="TimeUS") -> pd.DataFrame:
        """
        Returns an additional column to the dataframe that contains the relative time in units
        of seconds 

        Args:
            df (pd.DataFrame): The input dataframe to which the relative time column will be added.
            col (str): The name of the column containing the absolute time values.

        Returns:
            pd.DataFrame: The input dataframe with an additional column "t" containing the relative time in seconds.

        """
        if df is None or df.empty or col not in df:
            return df
        df = df.sort_values(col).copy()
        t0 = df[col].iloc[0]
        df["t"] = (df[col] - t0) / 1e6
        return df

    def get_desired_data(self,
                         types: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Args:
            types (List[str]): A list of message types to extract from the log.
            The names from the string come from the ardupilot documentation, 
            refer to https://ardupilot.org/plane/docs/common-downloading-and-analyzing-data-logs-in-mission-planner.html
            in the Message Details section of the link
        Returns:
            Dict[str, pd.DataFrame]: A dictionary of DataFrames, one for each message type.
        """
        if not types:
            raise ValueError("Types list cannot be empty.")

        rows: Dict[str, List[Dict[str, float]]] = {t: [] for t in types}
        while True:
            m = self.binary_log.recv_msg()
            if m is None:
                break
            t = m.get_type()
            if t in types:
                d = m.to_dict()
                d["TimeUS"] = getattr(m, "TimeUS", None)
                d["_t"] = getattr(m, "_timestamp", None)
                rows[t].append(d)

        dfs = {t: pd.DataFrame(rows[t]) for t in types}
        for k, df in dfs.items():
            dfs[k] = self.add_rel_time(df)

        return dfs


if __name__ == "__main__":
    flight_parser: FlightParser = FlightParser(log_path="data/example.BIN")

    """
    Some useful information about the types:
    NTUN (navigation information) -> provides 
    """
    desired_data: Dict[str, pd.DataFrame] = flight_parser.get_desired_data(
        types=["GPS", "NTUN", "CMD", "MODE", "IMU", "ATT", "AHR2"])

    print("desired_data:", desired_data)

    # Example of a plot
    attitude_data = desired_data.get("ATT", pd.DataFrame())
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(attitude_data["t"], attitude_data["Roll"], label="Roll")
    ax[0].plot(attitude_data["t"], attitude_data["DesRoll"],
               label="Desired Roll")
    ax[1].plot(attitude_data["t"], attitude_data["Pitch"], label="Pitch")
    ax[1].plot(attitude_data["t"], attitude_data["DesPitch"],
               label="Desired Pitch")
    ax[2].plot(attitude_data["t"], attitude_data["Yaw"], label="Yaw")
    ax[2].plot(attitude_data["t"], attitude_data["DesYaw"],
               label="Desired Yaw")
    for a in ax:
        a.legend()
        a.grid()
    plt.show()
