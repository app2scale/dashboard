from gevent import monkey
monkey.patch_all(thread=False, select=False, subprocess=False, ssl=False)
import solara
import time
from locust.env import Environment
from kubernetes import client, config
from ray.rllib.algorithms.algorithm import Algorithm
import ray
import numpy as np
import time
import pandas as pd
from itertools import product
from ray.rllib.algorithms import ppo, sac, dqn
import random
from gymnasium.spaces import Discrete, Box
import logging
from locust import HttpUser, task, constant, constant_throughput, events
from locust.shape import LoadTestShape
import math


WARM_UP_PERIOD = 300
# How many seconds to wait for transition
COOLDOWN_PERIOD = 0
# How many seconds to wait for metric collection
COLLECT_METRIC_TIME = 15
# Maximum number of metric collection attempt
COLLECT_METRIC_MAX_TRIAL = 200
# How many seconds to wait when metric collection fails
COLLECT_METRIC_WAIT_ON_ERROR = 2
# How many seconds to wait if pods are not ready
CHECK_ALL_PODS_READY_TIME = 2
# Episode length (set to batch size on purpose)
EPISODE_LENGTH = 100
PROMETHEUS_HOST_URL = "http://prometheus.local.payten.com.tr"

DEPLOYMENT_NAME = "teastore-webui"
NAMESPACE = "app2scale"

OBSERVATION_SPACE = Box(low=np.array([1, 4, 0, 0, 0, 0]), high=np.array([3, 9, 1, 1, 5, 5000]), dtype=np.float32)
ACTION_SPACE = Discrete(5)

number_of_rollout_workers = 0
number_of_gpus = 0

config_dqn = (
    dqn.DQNConfig()
    .environment(env=None, observation_space=OBSERVATION_SPACE, action_space=ACTION_SPACE)
    .rollouts(num_rollout_workers=number_of_rollout_workers, enable_connectors=False, num_envs_per_worker=1)
    .resources(num_gpus=number_of_gpus, num_cpus_per_worker=1)
    .training(train_batch_size=256,
                model={"fcnet_hiddens": [32,32]})
    # .callbacks(MetricCallbacks)
)
logging.getLogger().setLevel(logging.INFO)
previous_tps = 0
expected_tps = 1



class TeaStoreLocust(HttpUser):
    wait_time = constant_throughput(expected_tps)
    host = "http://teastore.local.payten.com.tr/tools.descartes.teastore.webui/"

    @task
    def load(self):
        self.visit_home()
        self.login()
        self.browse()
        # 50/50 chance to buy
        #choice_buy = random.choice([True, False])
        #if choice_buy:
        # self.buy()
        self.visit_profile()
        self.logout()

    def visit_home(self):

        # load landing page
        res = self.client.get('/')
        if res.ok:
            pass
        else:
            logging.error(f"Could not load landing page: {res.status_code}")

    def login(self):

        # load login page
        res = self.client.get('/login')
        if res.ok:
            pass
        else:
            logging.error(f"Could not load login page: {res.status_code}")
        # login random user
        user = random.randint(1, 99)
        login_request = self.client.post("/loginAction", params={"username": user, "password": "password"})
        if login_request.ok:
            pass
        else:
            logging.error(
                f"Could not login with username: {user} - status: {login_request.status_code}")
            

    def browse(self):

        # execute browsing action randomly up to 5 times
        for i in range(1, 2):
            # browses random category and page
            category_id = random.randint(2, 6)
            page = random.randint(1, 5)
            #page = 1
            category_request = self.client.get("/category", params={"page": page, "category": category_id})
            if category_request.ok:
                # logging.info(f"Visited category {category_id} on page 1")
                # browses random product
                product_id = random.randint((category_id-2)*100+7+(page-1)*20, (category_id-2)*100+26+(page-1)*20)
                product_request = self.client.get("/product", params={"id": product_id})
                if product_request.ok:
                    #logging.info(f"Visited product with id {product_id}.")
                    cart_request = self.client.post("/cartAction", params={"addToCart": "", "productid": product_id})
                    if cart_request.ok:
                        pass
                        #logging.info(f"Added product {product_id} to cart.")
                    else:
                        logging.error(
                            f"Could not put product {product_id} in cart - status {cart_request.status_code}")
                else:
                    logging.error(
                        f"Could not visit product {product_id} - status {product_request.status_code}")
            else:
                logging.error(
                    f"Could not visit category {category_id} on page 1 - status {category_request.status_code}")
                


    def buy(self):

        # sample user data
        user_data = {
            "firstname": "User",
            "lastname": "User",
            "adress1": "Road",
            "adress2": "City",
            "cardtype": "volvo",
            "cardnumber": "314159265359",
            "expirydate": "12/2050",
            "confirm": "Confirm"
        }
        buy_request = self.client.post("/cartAction", params=user_data)
        if buy_request.ok:
            pass
            # logging.info(f"Bought products.")
        else:
            logging.error("Could not buy products.")

    def visit_profile(self) -> None:

        profile_request = self.client.get("/profile")
        if profile_request.ok:
            pass
            # logging.info("Visited profile page.")
        else:
            logging.error("Could not visit profile page.")

    def logout(self):

        logout_request = self.client.post("/loginAction", params={"logout": ""})
        if logout_request.ok:
            pass
            # logging.info("Successful logout.")
        else:
            logging.error(f"Could not log out - status: {logout_request.status_code}")


class CustomLoad(LoadTestShape):

    trx_load_data = pd.read_csv("agent/data/transactions.csv")
    trx_load = trx_load_data["transactions"].values.tolist()
    trx_load = (trx_load/np.max(trx_load)*30).astype(int)+1
    indexes = [(177, 184), (661, 685), (1143, 1152), (1498, 1524), (1858, 1900)]
    clipped_data = []
    for idx in indexes:
        start, end = idx
        clipped_data.extend(trx_load[start:end+1])
    ct = 0
    # array1 = np.linspace(24, 144, 6, dtype=np.int32)/8
    # array2 = np.linspace(120, 24, 5, dtype=np.int32)/8

    # clipped_data = np.concatenate([array1, array2])
    clipped_data += np.random.randint(-1, 1, len(clipped_data))

    def tick(self):
        if self.ct >= len(self.clipped_data):
            self.ct = 0
        user_count = self.clipped_data[self.ct]

        return (user_count, user_count) 
    




test_plot_data = solara.reactive({'step': [], 'replica': [], 'cpu': [], "cpu_usage": [],"load": [], 
                                  "num_request": [], "response_time": []})


@solara.component
def Plot1D(x, y, title, legend, plot_number):

    series_list = []
    if plot_number > 1:
        for i in range(plot_number):
            series_list.append({
                "name": legend[i],
                "type": "line",
                "data": y[i]
            })
    else:
        series_list.append({
            "name": legend,
            "type": "line",
            "data": y
        })
    options = {
        "xAxis": {
            "type": "category",
            "data": x,
        },
        "yAxis": {
            "type": "value",
        },
        
        "series": series_list,
        "title": {
            "text": title,
            "left": "center"
        },
        "legend": {
            "orient": 'vertical',
            "right": 0,
            "data": legend,
        },
    }

    solara.FigureEcharts(option=options)


@solara.component
def StatusPlots(data):
    
    with solara.GridFixed(columns=3):
        # with solara.Column():
        Plot1D(data["step"], data["replica"], "Replica number", "replica", 1)
        Plot1D(data["step"], data["cpu"], "CPU limit", "cpu", 1)
        Plot1D(data["step"], data["cpu_usage"], "CPU usage", "cpu usage", 1)
        Plot1D(data["step"], [list(data["num_request"]),list(data["load"])], "Processed req/Load", ["processed req", "load"], 2)
        Plot1D(data["step"], data["response_time"], "Response time", "response time", 1)




def get_state(deployment):
    replica = deployment.spec.replicas
    cpu = int(int(deployment.spec.template.spec.containers[0].resources.limits["cpu"][:-1])/100)
    heap = int(int(deployment.spec.template.spec.containers[0].env[2].value[4:-1])/100)
    return np.array([replica, cpu]) # heap çıkarıldı

def get_deployment_info():
    config.load_kube_config()
    v1 = client.AppsV1Api()
    deployment = v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
    state = get_state(deployment)
    return deployment, state

def update_and_deploy_deployment_specs(target_state):
    
    deployment, _ = get_deployment_info()
    deployment.spec.replicas = int(target_state[0])
    deployment.spec.template.spec.containers[0].resources.limits["cpu"] = str(target_state[1]*100) + "m"

    config.load_kube_config()
    v1 = client.AppsV1Api()
    v1.patch_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE, deployment)


def get_running_pods():
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(NAMESPACE, label_selector=f"run={DEPLOYMENT_NAME}")
    running_pods = []
    for pod in pods.items:
        if pod.status.phase.lower() == "running" and pod.status.container_statuses[0].ready and pod.metadata.deletion_timestamp == None:
            running_pods.append(pod.metadata.name)
    return running_pods, len(pods.items)

def get_usage_metrics_from_server(running_pods_array):
    config.load_kube_config()
    api = client.CustomObjectsApi()
    k8s_pods = api.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", "app2scale", "pods")
    usage_metric_server = {}

    for stats in k8s_pods['items']:
        if stats["metadata"]["name"] in running_pods_array:
            try:
                usage_metric_server[stats["metadata"]["name"]] = [round(float(stats["containers"][0]["usage"]["cpu"].rstrip('n'))/1e9, 3),
                                                            round(float(stats["containers"][0]["usage"]["memory"].rstrip('Ki'))/(1024*1024),3)]
            except:
                usage_metric_server[stats["metadata"]["name"]] = [round(float(stats["containers"][0]["usage"]["cpu"].rstrip('n'))/1e9, 3),
                                            round(float(stats["containers"][0]["usage"]["memory"].rstrip('M'))/(1024),3)]

    usage_metric_server["cpu"], usage_metric_server["memory"] = np.mean(np.array(list(usage_metric_server.values()))[:,0]), np.mean(np.array(list(usage_metric_server.values()))[:,1])
    return usage_metric_server

def collect_metrics(env):
    deployment, state = get_deployment_info()
    while True:
        running_pods, number_of_all_pods = get_running_pods()
        if len(running_pods) == state[0] and state[0] == number_of_all_pods and running_pods:
            print("İnitial running pods", running_pods)
            break
        else:
            time.sleep(CHECK_ALL_PODS_READY_TIME)
    time.sleep(WARM_UP_PERIOD)
    env.runner.stats.reset_all()
    time.sleep(COLLECT_METRIC_TIME)
    n_trials = 0
    while n_trials < COLLECT_METRIC_MAX_TRIAL:
        print('try count for metric collection',n_trials)
        metrics = {}
        cpu_usage = 0
        memory_usage = 0

        empty_metric_situation_occured = False
        #running_pods, _ = get_running_pods()
        print("collect metric running pods", running_pods)
        try:
            metric_server = get_usage_metrics_from_server(running_pods)
            if metric_server["cpu"] and metric_server["memory"]:
                cpu_usage = metric_server["cpu"]
                memory_usage = metric_server["memory"]
            else:
                empty_metric_situation_occured = True
                break
        except Exception as e:
            print(e)
            

        if empty_metric_situation_occured:
            n_trials += 1
            time.sleep(COLLECT_METRIC_WAIT_ON_ERROR)
        else:
            #print("TEST", running_pods, len(running_pods))
            metrics['replica'] = state[0]
            metrics['cpu'] = state[1]
            metrics['heap'] = 7
            metrics["cpu_usage"] = cpu_usage
            metrics["memory_usage"] = memory_usage
            metrics['num_requests'] = round(env.runner.stats.total.num_requests/(COLLECT_METRIC_TIME + n_trials * COLLECT_METRIC_WAIT_ON_ERROR),2)
            metrics['num_failures'] = round(env.runner.stats.total.num_failures,2)
            metrics['response_time'] = round(env.runner.stats.total.avg_response_time,1)
            #print(env.runner.target_user_count, expected_tps)
            metrics['performance'] = min(round(metrics['num_requests'] /  (env.runner.target_user_count * expected_tps),6),1)
            metrics['expected_tps'] = env.runner.target_user_count * expected_tps*8 # 9 req for each user, it has changed now we just send request to the main page
            metrics['utilization'] = min(metrics["cpu_usage"]/(state[1]/10),1)
            print('metric collection succesfull')
            load.ct += 1
            return metrics
    return None

def step(action, state, env):
    global previous_tps
    print('Entering step function')
    if action == 0:
        temp_state = state + np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    elif action == 1: # increase_replica
        temp_state = state + np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif action == 2: # decrease_replica
        temp_state = state + np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif action == 3:
        temp_state = state + np.array([0, 1, 0, 0, 0, 0], dtype=np.float32)
    else:
        temp_state = state + np.array([0 , -1, 0, 0, 0, 0], dtype=np.float32)

    temp_state= temp_state.astype(np.float32)
    if OBSERVATION_SPACE.contains(temp_state):
        new_state = temp_state.copy()
        updated_state = new_state[:2]
        print('applying the state...')
        print("updated state", updated_state)
        # update_and_deploy_deployment_specs(updated_state)
        print('Entering cooldown period...')
        time.sleep(WARM_UP_PERIOD)
        print('cooldown period ended...')
        print('entering metric collection...')
        metrics = collect_metrics(env)
        new_state[2] = metrics["cpu_usage"]
        new_state[3] = np.minimum(metrics["num_requests"]/metrics["expected_tps"],1).round(3)
        new_state[4] = (previous_tps/metrics["expected_tps"]).round(3)
        new_state[5] = round(metrics["response_time"],1)
        print('updated_state', new_state)
        print('metrics collected',metrics)
        cpu_reward = -math.exp((1-metrics["utilization"]))
        performance_reward = -math.exp((1-metrics["performance"]))
        resp_reward = -math.exp(0.5*(math.log10(metrics["response_time"])- math.log10(50)))
        reward = 0.45*performance_reward + 0.2*resp_reward + 0.2*cpu_reward - 0.05*math.exp((state[1]-4)/(9-4)) - 0.1*math.exp((state[0]-1)/(3-1))
        if metrics is None:
            return new_state, None, None
        
    else:
        new_state = state.copy()
        print('entering metric collection...')
        metrics = collect_metrics(env)
        new_state[2] = round(metrics["cpu_usage"],3)
        new_state[3] = np.minimum(metrics["num_requests"]/metrics["expected_tps"],1).round(3)
        new_state[4] = (previous_tps/metrics["expected_tps"]).round(3)
        new_state[5] = round(metrics["response_time"],1)
        reward = -10
    return new_state, reward, metrics

load = CustomLoad()
env = Environment(user_classes=[TeaStoreLocust], shape_class=load) 

    
def start_demo():
    global previous_tps
    env.create_local_runner()
    env.runner.start_shape() 
    algo = config_dqn.build()
    checkpoint_path = "agent/checkpoints/checkpoint_010000"
    algo.restore(checkpoint_path)


    step_list = []
    replica_array = []
    cpu_array = []
    cpu_usage_array = []
    num_request_array = []
    load_array = []
    response_time_array = []
    _, prev_state = get_deployment_info()
    obs = prev_state.copy()
    obs = np.append(obs, np.array([0.3, 1, 1, 40]))
    for i in range(10):

        action = algo.compute_single_action(obs, explore=False)
        next_state, reward, metrics = step(action, obs, env)
        obs = next_state
        step_list.append(i)
        replica_array.append(obs[0])
        cpu_array.append(obs[1])
        cpu_usage_array.append(metrics["cpu_usage"])
        load_array.append(metrics["expected_tps"])
        response_time_array.append(metrics["response_time"])
        num_request_array.append(metrics["num_requests"])
        previous_tps = metrics["expected_tps"] 

        test_plot_data.set(
                {'step':step_list.copy(),
                 'replica': replica_array.copy(),
                 'cpu': cpu_array.copy(),
                 'cpu_usage': cpu_usage_array.copy(),
                 "load": load_array.copy(),
                 "response_time": response_time_array.copy(),
                 "num_request": num_request_array.copy()
                 })


    env.runner.quit()

@solara.component
def Page():
    solara.Button(label="Start", on_click=start_demo)
    StatusPlots(test_plot_data.value)
