from sumo_rl import SumoEnvironment
import traci
import os

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ

# the first phase in tls plan. see 'pedcrossing.tll.xml'
VEHICLE_GREEN_PHASE = 0
PEDESTRIAN_GREEN_PHASE = 0
# the id of the traffic light (there is only one). This is identical to the
# id of the controlled intersection (by default)
TLSID = 'C'
# pedestrian edges at the controlled intersection
WALKINGAREAS = [':C_w0', ':C_w1']
CROSSINGS = [':C_c0']

class SumoCfgEnvironment(SumoEnvironment):
    def __init__(self, net_file: str, route_file: str, cfg_file: str, out_csv_name: str, use_gui: bool):
        self._cfg = cfg_file
        super().__init__(net_file, route_file, out_csv_name, use_gui)

    def _start_simulation(self):
        sumo_cmd = [self._sumo_binary, '-c', self._cfg]

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

    def _get_waiting_ped(self):
        """
        Retrieve the number of peds with speed = 0 in every incoming lane
        """
        numWaiting = 0
        for edge in WALKINGAREAS:
            peds = self.sumo.edge.getLastStepPersonIDs(edge)
            for ped in peds:
                if (self.sumo.person.getWaitingTime(ped) == 1 and
                    self.sumo.person.getNextEdge(ped) in CROSSINGS):
                    numWaiting = self.sumo.trafficlight.getServedPersonCount(TLSID, PEDESTRIAN_GREEN_PHASE)
        return numWaiting
    
    def _get_system_info(self):
        info = super()._get_system_info()
        ped_waiting_time = self._get_waiting_ped()
        info['system_total_waiting_time'] += ped_waiting_time
        return info
    