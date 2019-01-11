import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import random
from dynamic_obstacles import DynaObs

class Map:
    def __init__(self,**kwargs):
        self.StartPoint = kwargs['start_point']
        self.TerminalPoint = kwargs['terminal']
        self.StaticObstacle = kwargs['sta_obstacle']
        try:
            self.DynamicObstacle = kwargs['dyn_obstacle']
            self.DynamicObstacle.initial(x_lim=[0, 10], y_lim=[0, 10], velocity_lim=[1, 0.5], method='random')
        except KeyError:
            self.DynamicObstacle = []
        self.obstacle = self.StaticObstacle + [tuple(term) for term in list(self.DynamicObstacle.ob_states[:, :2])]


    def update_obs(self):
        self.DynamicObstacle.update()
        self.obstacle = self.StaticObstacle + [tuple(term) for term in list(self.DynamicObstacle.ob_states[:, :2])]

class Robot:
    def __init__(self):
        self.global_time = 0
        self.interval_time = 0.1

        self.pos = (0, 0)
        self.radius = 0.3
        self.heading_dir = 0#math.pi/2
        self.theta_target = 0
        self.distance_target = 0

        self.D_max = 2.1  # the critical line of the safe region and unsafe region.
        self.SR = 0
        self.Vlmax = 4
        self.Vamax = 2
        self.dv_max = 2
        self.da_max = 1
        self.dt = 0.001 
        
        self.Vlmin_dw = 0
        self.Vlmax_dw = 0
        self.Vamin_dw = 0
        self.Vamax_dw = 0
        self.res_l = 0
        self.res_a = 0

        self.R = 0
        self.A = 0
        self.E = 0
        self.VL = 0
        self.VA = 0
        self.state = (0,0,0,0,0)

        self.Vl = 0   # current state linear and angular velocity.
        self.Va = 0

        self.Vl__ = 0.01  # calculated action to take.
        self.Va__ = 0.01

        self.vf = 1
        self.wf = 0.5

        self.FA = (self.heading_dir-math.pi/2, self.heading_dir+math.pi/2)

        self.Q = {}
        self.draw_list_x = []
        self.draw_list_y = []

    def read_q_table(self):
        infile = open('qtable.pickle','rb')
        self.Q = pickle.load(infile)
        infile.close()

    def __update_robot_state(self,current_time):
        self.interval_time = current_time - self.global_time
        self.global_time = current_time
        self.FA = (self.heading_dir - math.pi / 2, self.heading_dir + math.pi / 2)

    def init_robot(self, g_map):
        self.heading_dir = math.pi/2
        self.__set_init_pos(g_map)
        self.__get_distance_target(g_map)
        self.__get_theta_target(g_map)
        self.Vl = 0   
        self.Va = 0
        self.Vl__ = 0.01 
        self.Va__ = 0.01
        self.FA = (self.heading_dir - math.pi / 2, self.heading_dir + math.pi / 2)
        self.__get_state(g_map)

    def __set_init_pos(self, g_map):
        self.pos = g_map.StartPoint

    def __get_distance_target(self, g_map):
        dx = g_map.TerminalPoint[0] - self.pos[0]
        dy = g_map.TerminalPoint[1] - self.pos[1]
        self.distance_target = (dx**2 + dy**2)**(1/2)
        return self.distance_target

    def __is_win_state(self,g_map):
        if self.__get_distance_target(g_map) < self.radius:
            return True
        return False

    def __get_theta_target(self, g_map):
        sin_tar = (g_map.TerminalPoint[1] - self.pos[1]) / self.distance_target
        if sin_tar>1:
            sin_tar = 1
        if sin_tar<-1:
            sin_tar = -1
        dx = g_map.TerminalPoint[0] - self.pos[0]
        asin = math.asin(sin_tar)
        
        if asin>=0 and dx>=0:
             res = math.asin(sin_tar)
        elif asin>=0 and dx<0:
            res = math.pi - math.asin(sin_tar)
        elif asin<0 and dx>=0:
            res = 2*math.pi + math.asin(sin_tar)
        else:
            res = math.pi - math.asin(sin_tar)

        self.theta_target = res

    def __dist_2_robot(self,obst):
        dx = obst[0] - self.pos[0]
        dy = obst[1] - self.pos[1]
        return (dx**2+dy**2)**(1/2)

    def __test_obs_in_FA(self,obs):
        obs_dis = self.__dist_2_robot(obs)
        sin_obs = (obs[1]-self.pos[1]) / obs_dis
        theta_obs = math.asin(sin_obs)
        if theta_obs > self.FA[0] and theta_obs < self.FA[1]:
            return True
        return False

    def __get_nearest_obstacle(self, g_map):  
        obs_list_in_FA = [obs for obs in g_map.obstacle if self.__test_obs_in_FA(obs)]
        try:
            nearest_obs = min(obs_list_in_FA,key=lambda x: self.__dist_2_robot(x))
        except ValueError:
            return None
        return nearest_obs

    def __get_nearest_obs_dis(self,g_map):
        nearest_obs = self.__get_nearest_obstacle(g_map)
        if not nearest_obs:
            return 1000
        return self.__dist_2_robot(nearest_obs)

    def __get_obs_theta(self,obs):
        if obs:
            obs_dx = obs[0] - self.pos[0]
            obs_dy = obs[1] - self.pos[1]
            obs_tan = obs_dy/obs_dx
            obs_theta = math.atan(obs_tan)
            if obs_theta < 0:
                obs_theta = math.pi + obs_theta
            return obs_theta
        else:
            return 0

    def __is_fail_state(self,g_map):
        if self.__get_nearest_obs_dis(g_map) < self.radius:
            return True
        return False

    def __get_state_R(self,g_map):
        nearest_obs_dis = self.__get_nearest_obs_dis(g_map)
        if nearest_obs_dis <= self.D_max / 3:
            self.R = 0
            self.SR = 0
        elif nearest_obs_dis <= self.D_max * 2 / 3:
            self.R = 1
            self.SR = 0
        elif nearest_obs_dis <= self.D_max:
            self.R = 2
            self.SR = 0
        else:
            self.R = 3    # safe region
            self.SR = 1
    
    def __get_state_A(self,g_map):
        res = math.pi/10
        nearest_obs = self.__get_nearest_obstacle(g_map)
        try:
            nearest_obs_dis = self.__dist_2_robot(nearest_obs)
            cos_obs = (nearest_obs[0]-self.pos[0]) / nearest_obs_dis
            theta_obs = math.acos(cos_obs)
            alpha_obs = self.heading_dir - theta_obs
        except TypeError:
            alpha_obs = 0

        for i in range(10):
            if i * res > (math.pi/2-alpha_obs):
                self.A = i
                break

    def __get_state_E(self,g_map):
        res = math.pi/4
        alpha_target = self.heading_dir - self.theta_target
        if alpha_target >=0:
            dtheta = alpha_target
        else:
            dtheta = 2 * math.pi + alpha_target
        
        for i in range(8):
            if (i+1) * res >= dtheta:
                self.E = i
                break

    def __get_state_VL(self,g_map):
        res = self.Vlmax / 4
        for i in range(4):
            if (i+1)*res >= self.Vl:
                self.VL = i
                break
    
    def __get_state_VA(self,g_map):
        res = self.Vamax / 5
        for i in range(10):
            if (i+1)*res - self.Vamax >= self.Va:
                self.VA = i
                break

    def __get_state(self,g_map):
        self.__get_state_R(g_map)
        self.__get_state_A(g_map)
        self.__get_state_E(g_map)
        self.__get_state_VL(g_map)
        self.__get_state_VA(g_map)
        self.state = (self.R,self.A,self.E,self.VL,self.VA)
        self.FA = (self.heading_dir - math.pi / 2, self.heading_dir + math.pi / 2)

    def __dynamic_window(self):
        self.Vlmin_dw = max(self.Vl-self.dv_max*self.dt,0)
        self.Vamin_dw = max(self.Va-self.da_max*self.dt,-self.Vamax)
        self.Vlmax_dw = min(self.Vl+self.dv_max*self.dt,self.Vlmax)
        self.Vlmin_dw = min(self.Va+self.da_max*self.dt,self.Vamax)

        self.res_l = (self.Vlmax_dw-self.Vlmin_dw) / 4
        self.res_a = (self.Vamax_dw-self.Vamin_dw) / 10

    def __get_action_from_dw(self,n):
        self.__dynamic_window()
        Vnl = self.Vlmin_dw + math.floor((n-1)/4)*self.res_l
        Vna = self.Vamin_dw + ((n-1)%10)*self.res_a
        if abs(Vnl) < 0.01:
            Vnl = random.random() / 10
        return Vnl,Vna

    def __virtual_trajectory(self,g_map):
        px = self.pos[0]
        py = self.pos[1]
        heading_theta = self.heading_dir
        #print('vl:',self.Vl__,'va:',self.Va__,"heading dir: ",self.heading_dir)
        for i in range(math.floor(self.interval_time / self.dt)):
            px += self.Vl__*math.cos(heading_theta)*self.dt
            py += self.Vl__*math.sin(heading_theta)*self.dt
            heading_theta += self.Va__*self.dt
        #print("px,py:",px,py)
        self.draw_list_x.append(px)
        self.draw_list_y.append(py)

        g_map.update_obs()

        self.pos = (px,py)
        self.heading_dir = heading_theta
        self.Va = self.Va__
        self.Vl = self.Vl__
        self.__get_theta_target(g_map)
        self.__get_distance_target(g_map)
        self.FA = (self.heading_dir - math.pi / 2, self.heading_dir + math.pi / 2)

    def __reward_function(self,g_map):
        Dv = self.__get_nearest_obs_dis(g_map)
        alpha_v = self.heading_dir - self.theta_target
        self.__get_state(g_map)
        win = self.__is_win_state(g_map)
        fail = self.__is_fail_state(g_map)
        if win:
            return 2
        elif fail:
            return -1
        elif self.SR == 1:
            return 1
        else:
            return 0.05 * Dv + 0.2 * alpha_v + 0.1 * self.Vl__

    def init_Q(self):
        for r in range(3):
            for a in range(10):
                for e in range(8):
                    for vl in range(4):
                        for va in range(10):
                            self.Q[(r,a,e,vl,va)] = [0] * 40


    def __SR_judge(self,g_map,dis2Ter,dis2Obs,theta2Ter,theta2Obs):
        vdis2T = self.__dist_2_robot(g_map.TerminalPoint)-dis2Ter              # the more negative the better
        dtheta_tar = abs(theta2Ter-self.theta_target)                              # the more close to 0 the better
        nearest_obs = self.__get_nearest_obstacle(g_map)
        nearest_obs_dis = self.__get_nearest_obs_dis(g_map)
        if nearest_obs:
            obs_theta = self.__get_obs_theta(nearest_obs)
            dtheta_obs = abs(theta2Obs-obs_theta)             # the most important element: varation on distance to the nearest obstacle. We should get over the obstacle.
            vdis2O = nearest_obs_dis - dis2Obs                # so dtheta_obs the larger, the better. vdis2O the larger the better
        else:
            vdis2O = 0
            dtheta_obs = 0

        #print("vary on dis to terminal:",vdis2T,"dtheta:",dtheta)
        return  vdis2O * 2 + dtheta_obs*20 + vdis2T * -4 + dtheta_tar * -1

        # vdis2Tar = self.__dist_2_robot(g_map.TerminalPoint) - dis2Ter
        # nearest_obs_dis = self.__get_nearest_obs_dis(g_map)
        # vdis2Obs = nearest_obs_dis - dis2Obs
        # res = vdis2Tar*(-4) + vdis2Obs * 10
        # print(res)
        # return res


    def __gen_heur_list(self,n):
        vl,va = self.__get_action_from_dw(n)
        dv = abs(self.Vl - vl)
        da = abs(self.Va - va)
        return dv+da*2

    def __policy_run(self,g_map):
        while True:
            current_dis2target = self.__dist_2_robot(g_map.TerminalPoint)
            current_theta = self.heading_dir
            current_dis2obs = self.__get_nearest_obs_dis(g_map)
            current_theta2Obs = self.__get_obs_theta(self.__get_nearest_obstacle(g_map))
            self.__get_state(g_map)
            current_state = self.state
            print(current_state)
            if self.SR == 1:
                self.Vl__ = self.vf
                if abs(self.heading_dir - self.theta_target)>0.2:
                    self.Va__ = self.wf
                else:
                    self.Va__ = 0

                state_mem = (self.pos,self.heading_dir)
                self.__virtual_trajectory(g_map)
                self.__get_state(g_map)

                judge1 = self.__SR_judge(g_map,current_dis2target,current_dis2obs,current_theta,current_theta2Obs)
                self.pos,self.heading_dir = state_mem
                self.Va__ = -self.Va__
                self.__virtual_trajectory(g_map)

                judge2 = self.__SR_judge(g_map,current_dis2target,current_dis2obs,current_theta,current_theta2Obs)
                
                if judge1 > judge2:
                    self.pos,self.heading_dir = state_mem
                    self.Va__ = -self.Va__
                    self.__virtual_trajectory(g_map)

                # dy = g_map.TerminalPoint[1]-self.pos[1]
                # dx = g_map.TerminalPoint[0]-self.pos[0]
                # dis = self.__dist_2_robot(g_map.TerminalPoint)
                # sin_tar = dy/dis
                # asin = math.asin(dy/dis)
                # if asin >= 0 and dx >= 0:
                #     res = math.asin(sin_tar)
                # elif asin >= 0 and dx < 0:
                #     res = math.pi - math.asin(sin_tar)
                # elif asin < 0 and dx >= 0:
                #     res = 2 * math.pi + math.asin(sin_tar)
                # else:
                #     res = math.pi - math.asin(sin_tar)
                # self.heading_dir = res
                # self.Va__ = 0
                # self.Vl__ = 2
                # self.__virtual_trajectory(g_map)

                print("Safe region,pos:",self.pos)
          
            else:
                heuristic_list = [self.__gen_heur_list(x) for x in range(40)]
                # heuristic_list[heuristic_list.index(min(heuristic_list))] = 9999

                while True:
                    act_idx = heuristic_list.index(max(heuristic_list))
                    state_mem = (self.pos,self.heading_dir)
                    vl_mem,va_mem = self.Vl,self.Va
                    current_obs_dis = self.__get_nearest_obs_dis(g_map)
                    current_obs_theta = self.__get_obs_theta(self.__get_nearest_obstacle(g_map))
                    if current_obs_dis > self.distance_target:
                        self.Vl__ = self.vf
                        if abs(self.heading_dir - self.theta_target)>0.2:
                            self.Va__ = self.wf
                        else:
                            self.Va__ = 0

                        state_mem = (self.pos,self.heading_dir)
                        self.__virtual_trajectory(g_map)
                        self.__get_state(g_map)

                        judge1 = self.__SR_judge(g_map,current_dis2target,current_obs_dis,current_theta,current_obs_theta)
                        self.pos,self.heading_dir = state_mem
                        self.Va__ = -self.Va__
                        self.__virtual_trajectory(g_map)

                        judge2 = self.__SR_judge(g_map,current_dis2target,current_obs_dis,current_theta,current_obs_theta)

                        if judge1 > judge2:
                            self.pos,self.heading_dir = state_mem
                            self.Va__ = -self.Va__
                            self.__virtual_trajectory(g_map)
                        break
                    self.Vl__,self.Va__ = self.__get_action_from_dw(act_idx)
                    self.__virtual_trajectory(g_map)
                    self.__get_state(g_map)
                    if self.__SR_judge(g_map,current_dis2target,current_obs_dis,current_theta,current_obs_theta)<0 or self.__get_nearest_obs_dis(g_map)<current_obs_dis:
                        self.pos,self.heading_dir = state_mem
                        self.Vl, self.Va = vl_mem, va_mem
                        heuristic_list[act_idx] = -9999
                        if len(set(heuristic_list)) == 1:
                            #heuristic_list = [self.__gen_heur_list(x) for x in range(40)]
                            #act_idx = heuristic_list.index(min(heuristic_list))
                            act_idx = random.randint(0,39)
                            self.Vl__,self.Va__ = self.__get_action_from_dw(act_idx)
                            self.__virtual_trajectory(g_map)
                            self.__get_state(g_map)
                            break
                    else:
                        break

                self.Q[current_state][act_idx] = self.__reward_function(g_map) 

                print("Unsafe region,pos:",self.pos)
                
                if self.__is_fail_state(g_map):
                    break
            if self.__is_win_state(g_map):
                break

    def train(self,g_map,scenario=100):
        sce_num = scenario
        for i in range(sce_num):
            self.__policy_run(g_map)
            print("scenario:",i)
            self.init_robot(g_map)

    def draw_train_track(self,g_map):
        plt.scatter(self.draw_list_x,self.draw_list_y)
        obsx = []
        obsy = []
        for obs in g_map.obstacle:
            obsx.append(obs[0])
            obsy.append(obs[1])
        plt.scatter(obsx,obsy)
        plt.show()

    def run(self,g_map):
        current_dis2target = self.__dist_2_robot(g_map.TerminalPoint)
        current_theta = self.heading_dir
        current_obs_dis = self.__get_nearest_obs_dis(g_map)
        current_obs_theta = self.__get_obs_theta(self.__get_nearest_obstacle(g_map))
        self.__get_state(g_map)
        current_state = self.state
        print(current_state)

        if self.SR == 1:
            self.Vl__ = self.vf
            if abs(self.heading_dir - self.theta_target) > 0.2:
                self.Va__ = self.wf
            else:
                self.Va__ = 0

            state_mem = (self.pos, self.heading_dir)
            self.__virtual_trajectory(g_map)
            self.__get_state(g_map)

            judge1 = self.__SR_judge(g_map, current_dis2target,current_obs_dis,current_theta,current_obs_theta)
            self.pos, self.heading_dir = state_mem
            self.Va__ = -self.Va__
            self.__virtual_trajectory(g_map)

            judge2 = self.__SR_judge(g_map, current_dis2target,current_obs_dis,current_theta,current_obs_theta)

            if judge1 > judge2:
                self.pos, self.heading_dir = state_mem
                self.Va__ = -self.Va__
                self.__virtual_trajectory(g_map)
        else:
            Q_line = self.Q[current_state]
            act_idx = Q_line.index(max(Q_line))
            self.Vl__, self.Va__ = self.__get_action_from_dw(act_idx)
            self.__virtual_trajectory(g_map)
            self.__get_state(g_map)
                


    
    def save_q_table(self):
        outfile = open('qtable.pickle','wb')
        pickle.dump(self.Q,outfile)
        outfile.close()



if __name__ == "__main__":
    sta_obst = [(3,3),(4,4),(8,8)]
    world_map = Map(start_point=(0,0),terminal=(10,10),sta_obstacle=sta_obst, dyn_obstacle=DynaObs(2))
    robot1 = Robot()
    robot1.init_robot(world_map)
    robot1.init_Q()
    robot1.read_q_table()
    robot1.train(world_map, 1)
    robot1.draw_train_track(world_map)
    robot1.save_q_table()


