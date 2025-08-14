from locust import HttpUser, task, between
import random
import json

SAMPLE_PAYLOAD = {
    'impression_id': '1',
    'user_id': 'u1',
    'features': {}
}

class RouterUser(HttpUser):
    wait_time = between(0.01, 0.1)

    @task
    def predict(self):
        p = SAMPLE_PAYLOAD.copy()
        # mutate some fields to simulate variety
        p['impression_id'] = str(random.randint(1, 1000000))
        p['user_id'] = 'u' + str(random.randint(1, 10000))
        p['features'] = {'cat1': str(random.randint(0, 100)), 'num1': random.random()}
        self.client.post('/predict', json=p)