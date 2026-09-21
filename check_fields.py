from FlightParser import FlightParser

log = "/home/dronie/ardupilot_parse/data/landings/expert/00000042.BIN"
fp = FlightParser(log, verbose=True)
msg_types = ['XKF1', 'XKF2', 'XKF3', 'NTUN', 'ATT', 'AHR2', 'RCOU', 'RCIN', 'CMD', 'CTUN', 'IMU', 'GPS']
desired = fp.get_desired_data(msg_types)
for name in msg_types:
    df = desired.get(name)
    print(name, 'rows=', len(df) if df is not None else None)
    if df is not None and not df.empty:
        print('columns=', df.columns[:20].tolist())
        print(df.head(2).to_dict(orient='records')[0])
        break
