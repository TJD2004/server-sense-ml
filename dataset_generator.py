import math
import random
import csv
import os

def generate_solar_dataset(filename="solar_dataset.csv", num_samples=6000):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    random.seed(42)
    
    headers = [
        "hour",
        "month",
        "temp",
        "irradiance",
        "cloudCoverage",
        "humidity",
        "windSpeed",
        "capacityKW",
        "solar"
    ]
    
    rows = []
    
    for _ in range(num_samples):
        # Random feature sampling across realistic physical bounds
        hour = random.randint(6, 19)
        month = random.randint(1, 12)
        temp = round(random.uniform(15.0, 48.0), 1)
        
        # Clear sky sun angle curve
        peak = 12.5
        spread = 3.4
        clear_sky_base = 5.1 * math.exp(-pow(hour - peak, 2) / (2 * spread * spread))
        
        # Cloud coverage and weather variations
        cloudCoverage = round(random.uniform(0.0, 100.0), 1)
        cloudMult = max(0.05, 1.0 - (cloudCoverage / 100.0) * 0.88)
        
        # Ambient & Panel Temperature factor (0.4% efficiency drop per degree C above 25°C)
        tempMult = max(0.35, 1.0 - max(0.0, temp - 25.0) * 0.012)
        
        # Irradiance (W/m2)
        base_irr = max(50.0, min(1100.0, (clear_sky_base / 5.1) * 1000.0 * cloudMult))
        irradiance = round(max(50.0, base_irr + random.uniform(-40.0, 40.0)), 1)
        irrMult = irradiance / 1000.0
        
        # Humidity & Wind
        humidity = round(random.uniform(20.0, 95.0), 1)
        humidityMult = max(0.85, 1.0 - max(0.0, humidity - 60.0) * 0.0025)
        
        windSpeed = round(random.uniform(1.0, 30.0), 1)
        windCoolingMult = 1.0 + min(0.06, windSpeed * 0.002) # Wind cools hot panels
        
        # Panel capacity (kW)
        capacityKW = round(random.choice([3.0, 5.0, 7.5, 10.0]), 1)
        
        # Calculate target generation (kW)
        base_solar = capacityKW * (clear_sky_base / 5.1) * cloudMult * tempMult * irrMult * humidityMult * windCoolingMult
        
        # Add realistic sensor noise (0.05 - 0.15 kW wobble)
        noise = random.gauss(0.0, 0.08)
        solar = max(0.0, round(base_solar + noise, 2))
        
        rows.append([
            hour,
            month,
            temp,
            irradiance,
            cloudCoverage,
            humidity,
            windSpeed,
            capacityKW,
            solar
        ])
        
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"[OK] Successfully generated {len(rows)} samples in '{filename}'")

if __name__ == "__main__":
    generate_solar_dataset("ml_service/solar_dataset.csv", 6000)
