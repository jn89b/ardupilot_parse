import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
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


def pickle_desired_data(desired_data: Dict[str, pd.DataFrame], file_path: str) -> None:
    """
    Pickles the desired data dictionary to a specified file path.
    Data is saved as hashtables of dataframes for each message type.
    Args:
        desired_data (Dict[str, pd.DataFrame]): The dictionary of DataFrames to pickle.
        file_path (str): The file path where the pickled data will be saved.
    Returns:
        None
    """
    with open(file_path, 'wb') as f:
        pickle.dump(desired_data, f)


def generate_command_data(attitude_data: pd.DataFrame,
                          throttle_data: pd.DataFrame,
                          dt: float = 0.05,
                          file_name: str = "command_data",
                          save_to_csv: bool = False) -> pd.DataFrame:
    """
    Generates a roll,pitch,yaw,throttle command dataframe based on the attitude and throttle data.

    Args:
        attitude_data (pd.DataFrame): DataFrame containing 't', 'DesRoll', '
            'DesPitch', 'DesYaw' columns.
        throttle_data (pd.DataFrame): DataFrame containing 't', 'ThO' columns
        dt (float): Time step for the command data.
        file_name (str): Name of the file to save the command data CSV.
        save_to_csv (bool): Whether to save the generated command data to a CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the generated command data with columns
            't', 'DesRoll', 'DesPitch', 'DesYaw', 'DesThrottle'.

    """
    len_data = len(attitude_data)
    time_command = np.arange(0, len_data * dt, dt)
    desired_roll = np.interp(
        attitude_data['t'], attitude_data['t'], attitude_data['DesRoll'])
    desired_pitch = np.interp(
        attitude_data['t'], attitude_data['t'], attitude_data['DesPitch'])
    desired_yaw = np.interp(
        attitude_data['t'], attitude_data['t'], attitude_data['DesYaw'])
    desired_throttle = np.interp(
        throttle_data['t'], throttle_data['t'], throttle_data['ThO'])
    # map the throttle based on time
    mapped_throttle = np.interp(
        time_command, throttle_data['t'], desired_throttle)
    scaled_throttle = mapped_throttle / 100.0  # reduce to 0 to 1 range
    # regularize to the attitude time stamps
    command_data = pd.DataFrame({
        't': time_command,
        'DesRoll': desired_roll,
        'DesPitch': desired_pitch,
        'DesYaw': desired_yaw,
        'DesThrottle': scaled_throttle
    })
    #
    if save_to_csv:
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # folder = os.path.join(current_dir, "maneuvers")
        # os.makedirs(folder, exist_ok=True)
        # filepath = os.path.join(folder, str(file_name))
        filepath = file_name + "_parsed"
        command_data.to_csv(filepath + ".csv", index=False)

    return command_data


def sync_imu_rcou(
    imu_data: pd.DataFrame,
    rcou_data: pd.DataFrame,
    time_col: str = "TimeUS",
    direction: str = "nearest",
) -> pd.DataFrame:
    """
    Synchronize IMU and RCOU dataframes on time using merge_asof.

    direction: 'nearest', 'backward', or 'forward' for how to match times.
    """

    # If either is empty, just bail out cleanly
    if imu_data.empty or rcou_data.empty:
        print("One of the datasets is empty; returning empty DataFrame.")
        return pd.DataFrame()

    # Work on copies to avoid mutating original
    imu = imu_data.copy()
    rcou = rcou_data.copy()

    # Make sure the time column exists
    if time_col not in imu.columns or time_col not in rcou.columns:
        raise KeyError(
            f"'{time_col}' must be a column in both IMU and RCOU dataframes.")

    # Convert TimeUS (microseconds) to seconds and normalize to start at 0
    imu["t_s"] = (imu[time_col] - imu[time_col].iloc[0]) * 1e-6
    rcou["t_s"] = (rcou[time_col] - rcou[time_col].iloc[0]) * 1e-6

    imu = imu.drop_duplicates(subset="t_s", keep="first")
    rcou = rcou.drop_duplicates(subset="t_s", keep="first")

    # Sort by time (required for merge_asof)
    imu = imu.sort_values("t_s").reset_index(drop=True)
    rcou = rcou.sort_values("t_s").reset_index(drop=True)

    # Merge: each IMU row gets the nearest RCOU row in time
    synced = pd.merge_asof(
        imu,
        rcou,
        on="t_s",
        direction=direction,
        suffixes=("_imu", "_rcou"),
    )

    # skip
    synced.drop_duplicates(keep="first", subset=["t_s"], inplace=False)

    return synced


def make_csv(file_name: str) -> None:
    """
    Turn this into a parser to generate the file we want for live replay
    """
    flight_parser: FlightParser = FlightParser(log_path=file_name + ".BIN")
    desired_data: Dict[str, pd.DataFrame] = flight_parser.get_desired_data(
        types=["GPS", "XKF1", "CTUN", "MODE", "IMU", "ATT", "AHR2", "RCIN", "RCOU", "CMD", "ARSP"])
    attitude_data = desired_data.get("ATT", pd.DataFrame())
    throttle_data = desired_data.get("CTUN", pd.DataFrame())

    # IMU data, RCOU data
    imu_data = desired_data.get("IMU", pd.DataFrame())
    rcou_data = desired_data.get("RCOU", pd.DataFrame())
    kf_data = desired_data.get("XKF1", pd.DataFrame())
    kf_data.to_csv(file_name + "_kf_data.csv", index=False)

    # imu_data.to_csv(file_name + "_imu_data.csv", index=False)
    # rcou_data.to_csv(file_name + "_rcou_data.csv", index=False)

    # synched_data = sync_imu_rcou(
    #     imu_data=imu_data,
    #     rcou_data=rcou_data,
    #     time_col="TimeUS",
    #     direction="nearest",
    # )

    # synched_data.to_csv(file_name + "_synced_imu_rcou_data.csv", index=False)

    # command: pd.DataFrame = generate_command_data(
    #     attitude_data=attitude_data,
    #     throttle_data=throttle_data,
    #     dt=0.05,
    #     file_name=file_name+"_command_data",
    #     save_to_csv=True
    # )


if __name__ == "__main__":
    # make_csv("data/00000005")
    df = pd.read_csv("data/00000005_kf_data.csv")
    # slice the data to the first 1000 rows
    df = df.iloc[350:]
    # resave

    """
    The data you want to extract from the Kalman filter log includes:
    - Time (t): The timestamp of each data point.
    - Position (PN, PE, PD): The North, East, and Down position coordinates
        representing the vehicle's location in 3D space.
    - Velocity (VN, VE, VD): The North, East, and Down velocity components
        indicating the vehicle's speed in each direction.
    - Attitude (Roll, Pitch, Yaw): The orientation of the vehicle in terms
        of roll, pitch, and yaw degrees.
    """

    df.to_csv("data/00000005_kf_data_slice.csv", index=False)
    # plot 3d
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(df['PN'], df['PE'], df['PD'])
    ax.set_title("3D Position from Kalman Filter")
    ax.set_xlabel("PN (m)")
    ax.set_ylabel("PE (m)")
    ax.set_zlabel("PD (m)")

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
    ax.plot(df['t'], df['Roll'])

    # load the csv and compute the derivates of time
    # command_data = pd.read_csv("data/00000005_synced_imu_rcou_data.csv")
    # # command_data['dt'] = command_data['t_s'].diff().fillna(0)

    # t_imu = command_data['t_s'].values
    # dt_imu = np.diff(t_imu, prepend=t_imu[0])

    # # plot as a historgram of dt
    # import seaborn as sns
    # fig, ax = plt.subplots()
    # sns.histplot(dt_imu, bins=50, ax=ax)
    # ax.set_title("Histogram of IMU time differences")
    # ax.set_xlabel("Time difference (s)")

    plt.show()
