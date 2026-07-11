import numpy as np
import pandas as pd

from scipy.integrate import quad
from scipy.optimize import minimize
from datetime import datetime as dt

from scipy.stats import norm
from scipy.optimize import brentq



import pickle

with open("all_svi_params.pkl", "rb") as f:
    all_svi_params = pickle.load(f)

# BS Functions

def bs_price(S, K, T, r, q, sigma, option_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)
    else:
        return np.nan    

def no_arb_bounds(S, K, T, r, q, option_type):
    if option_type == "call":
        lower = max(S * np.exp(-q*T) - K * np.exp(-r*T), 0)
        upper = S * np.exp(-q*T)
    else:
        lower = max(K * np.exp(-r*T) - S * np.exp(-q*T), 0)
        upper = K * np.exp(-r*T)
    return lower, upper


def implied_vol(price, S, K, T, r, q, option_type, lo=1e-6, hi=5.0):
    lower, upper = no_arb_bounds(S, K, T, r, q, option_type)

    # Price outside arbitrage bounds cannot produce a valid Black-Scholes IV.
    if not (lower <= price <= upper):
        return np.nan

    def f(sig):
        return bs_price(S, K, T, r, q, sig, option_type) - price

    try:
        return brentq(f, lo, hi, maxiter=100)
    except ValueError:
        return np.nan
# Delta

def delta_call(S, K, T, sigma, r = 0.04, q = 0):
    
    d1 = ( np.log(S / K) + (r + q + 0.5 * sigma**2) * T ) / (sigma * np.sqrt(T))

    return np.exp(-q*T) * norm.cdf(d1)

def delta_put(S, K, T, sigma, r = 0.04, q = 0):

    d1 = ( np.log(S / K) + (r + q + 0.5 * sigma**2) * T ) / (sigma * np.sqrt(T))

    return np.exp(-q*T) * (norm.cdf(d1) - 1)


# Vega

def vega(S, K, T, sigma, r=0.04, q=0.0):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    d1 = (
        np.log(S / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


# Define SVI smile

def svi(k,a,b,rho,m,sigma):
    w = a + b * ( rho*(k-m) + np.sqrt((k-m)**2 + sigma** 2))
    return w

def svi_iv(date, K, S, T_option, r=0.04, q=0.0):
    K = float(K)
    S = float(S)
    T_option = float(T_option)

    if T_option <= 0:
        return np.nan

    T_available = np.array(sorted(all_svi_params[date].keys()), dtype=float)

    F = S * np.exp((r - q) * T_option)
    k = np.log(K / F)

    # if below shortest maturity, use shortest
    if T_option <= T_available[0]:
        T0 = T_available[0]
        w = svi(k, *all_svi_params[date][T0])
        return np.sqrt(w / T0)

    # if above longest maturity, use longest
    if T_option >= T_available[-1]:
        T0 = T_available[-1]
        w = svi(k, *all_svi_params[date][T0])
        return np.sqrt(w / T0)

    # find two surrounding maturities
    idx = np.searchsorted(T_available, T_option)

    T_low = T_available[idx - 1]
    T_high = T_available[idx]

    w_low = svi(k, *all_svi_params[date][T_low])
    w_high = svi(k, *all_svi_params[date][T_high])

    # interpolate total variance
    alpha = (T_option - T_low) / (T_high - T_low)

    w_interp = (1 - alpha) * w_low + alpha * w_high

    if w_interp <= 0:
        return np.nan

    return np.sqrt(w_interp / T_option)

def svi_price(date, K, S, T, r = 0.04, q =0, option_type = "call"):

    sigma = svi_iv(date, K, S, T)

    price = bs_price(S,K,T,r,q, sigma, option_type="call")

    return price

# Heston

with open("heston_params.pkl", "rb") as f:
    heston_params = pickle.load(f)

def heston_option(S0, K, r, t, kappa, v0, xi, rho, theta, option_type = "call"):
    """
    Price of a call option under Heston model
    
    Parameters:
    - S0 (float): Initial stock price
    - K (float): Strike Price
    - r (float): Risk-free interest rate
    - t (float): Time-to-expiration (in years)
    - v0 (float): Initial variance
    - kappa (float): Rate of mean reversion of variance (1 to 5)
    - theta (float): Long-run variance
    - xi (float): Volatility of volatility (.2 to 1)
    - rho (float): Correlation between Brownian motions (-.9 to -.2)
    - option_type (string): 'call' for call and 'put' for put

    Returns:
    - price (float): Option price
    """

    def integrand(phi, Pnum):
        i = complex(0, 1)
        u = 0.5 if Pnum == 1 else -0.5
        b = kappa - rho * xi if Pnum == 1 else kappa
        a = kappa * theta
        d = np.sqrt((rho * xi * phi * i - b)**2 - xi**2 * (2 * u * phi * i - phi**2))
        g = (b - rho * xi * phi * i + d) / (b - rho * xi * phi * i - d)
        
        exp1 = np.exp(i * phi * np.log(S0 / K))
        C = r * phi * i * t + a / xi**2 * ((b - rho * xi * phi * i + d) * t - 2 * np.log((1 - g * np.exp(d * t)) / (1 - g)))
        D = (b - rho * xi * phi * i + d) / xi**2 * ((1 - np.exp(d * t)) / (1 - g * np.exp(d * t)))
        f = exp1 * np.exp(C + D * v0)
        return np.real(f / (phi * i))

    P1 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 1), 0, 100)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 2), 0, 100)[0]
    call_price = S0 * P1 - K * np.exp(-r * t) * P2
    put_price = call_price - S0 + K * np.exp(-r * t)
    
    if option_type == 'call':
        return call_price
    
    if option_type == 'put':
        return put_price
   

def heston_iv(date, S, K, t, r = 0.04):
    
    kappa, v0, xi, rho, theta = heston_params[date]

    price = heston_option(S, K, t, r, kappa, v0, xi, rho, theta)

    iv = implied_vol(price, S, K, t, r,  q = 0, option_type = "call")

    return iv



# NN




import torch
import torch.nn as nn

class VolSurfaceNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
    



# Load everything once
all_nn_objects = torch.load(
    "all_nn_objects.pt",
    map_location="cpu",
    weights_only=False
)

all_nn_models = {}

for date, obj in all_nn_objects.items():
    model = VolSurfaceNN()
    model.load_state_dict(obj["state_dict"])
    model.eval()

    all_nn_models[date] = {
        "model": model,
        "x_scaler": obj["x_scaler"],
        "y_scaler": obj["y_scaler"]
    }

def nn_iv(date,k, T, S0, is_call = True):
    """
    Predict implied volatility from the trained neural network.

    Parameters
    ----------
    k : float
        Log-moneyness
    T : float
        Time to maturity (years)
    S0 : float
        Current underlying price
    is_call : int
        1 for call, 0 for put

    Returns
    -------
    float
        Predicted implied volatility
    """

    obj = all_nn_models[date]

    model = obj["model"]
    x_scaler = obj["x_scaler"]
    y_scaler = obj["y_scaler"]
    

    if is_call == True:
        x = np.array([[k, T, 1]])
    else: x = np.array([k,T,0])
    
    x = x_scaler.transform(x)
    x = torch.tensor(x, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred = model(x).numpy()

    w = max(y_scaler.inverse_transform(pred)[0, 0], 1e-10)
    T = max(T,1e-8)
    iv = np.sqrt(w / T)

    return float(iv)