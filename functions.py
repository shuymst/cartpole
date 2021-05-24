import numpy as np
def feature(state, dim):
    cart_p, cart_v, pole_a, pole_v = state
    if dim == "quad":
        return np.array([cart_p*cart_p,cart_v*cart_v,pole_a*pole_a,pole_v*pole_v,cart_p*cart_v,cart_p*pole_a,cart_p*pole_v,cart_v*pole_a,cart_v*pole_v,pole_a*pole_v,cart_p,cart_v,pole_a,pole_v])
    elif dim == "linear":
        return np.array([cart_p,cart_v,pole_a,pole_v])
    elif dim == "mid":
        return np.array([cart_p,cart_v,pole_a,pole_v,cart_p*cart_v,pole_a*pole_v])
    
def make_x(state, action, dim):
    num = feature(state, dim).shape
    a = np.zeros(num)
    if action == 0:
        return np.append(feature(state, dim), a)
    else:
        return np.append(a, feature(state, dim))

def preference(theta, state, dim):
    preferences = np.zeros(2)
    for action in [0, 1]:
        x = make_x(state, action, dim)
        h = np.dot(theta, x)
        preferences[action] = h
    return preferences

def softmax(preferences):
    c = np.max(preferences)
    exp_preferences = np.exp(preferences - c)
    return exp_preferences / np.sum(exp_preferences)

def get_action(theta, state, dim):
    preferences = preference(theta, state, dim)
    action_probs = softmax(preferences)
    return np.random.choice([0, 1], p = action_probs)

def update(theta, state, action, g, alpha, dim):
    diff_log = make_x(state, action, dim)
    preferences = preference(theta, state, dim)
    action_probs = softmax(preferences)
    for b in [0,1]:
        diff_log -= action_probs[b] * make_x(state, b, dim)
    return theta + alpha * g * diff_log