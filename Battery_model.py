

import numpy as np

class Battery:
    def __init__(self, capacity_kwh, max_power_kw, 
                 charge_efficiency, discharge_efficiency, 
                 initial_soc = 0.5,):
        
        if capacity_kwh <= 0 or max_power_kw <= 0 or charge_efficiency <= 0 or discharge_efficiency <= 0 \
            or charge_efficiency > 1 or discharge_efficiency > 1 or initial_soc < 0 or initial_soc > 1:

            raise ValueError("Invalid battery parameters, all parameters must be positive and efficiencies must be between 0 and 1.")
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.soc = initial_soc
    
    @property
    def soc_kwh(self):
        """current energy in kWh based on SoC and capacity"""
        return self.soc * self.capacity_kwh
    
    def charge(self, power_kw_bought, duration_hours):
        if not np.isfinite(duration_hours):
            raise ValueError("Duration must be finite.")
        if not np.isfinite(power_kw_bought):
            raise ValueError("Power bought must be finite.")
        if duration_hours <= 0:
            raise ValueError("Duration must be positive.")
        if power_kw_bought <= 0:
            raise ValueError("Power bought must be positive.")
        if power_kw_bought   > self.max_power_kw:
            power_kw_bought = self.max_power_kw

        grid_in_kwh = power_kw_bought * duration_hours
        
        battery_gain_kwh = grid_in_kwh * self.charge_efficiency
        
        remaining_space_kwh = self.capacity_kwh - self.soc_kwh
        battery_gain_kwh = min(battery_gain_kwh, remaining_space_kwh)
        
        # Recompute grid_in_kwh = battery_gain_kwh / charging efficiency 
        grid_in_kwh = battery_gain_kwh / self.charge_efficiency
        
        # Update SoC using battery_gain_kwh
        self.soc += battery_gain_kwh / self.capacity_kwh
        self.soc = min(max(self.soc, 0.0), 1.0)
        
        return grid_in_kwh
    
    def discharge(self, power_kw_sold, duration_hours):
        if not np.isfinite(duration_hours):
            raise ValueError("Duration must be finite.")
        if not np.isfinite(power_kw_sold):
            raise ValueError("Power sold must be finite.")
        if duration_hours <= 0:
            raise ValueError("Duration must be positive.")
        if power_kw_sold <= 0:
            raise ValueError("Power sold must be positive.")
        if power_kw_sold > self.max_power_kw:
            power_kw_sold = self.max_power_kw

        grid_out_kwh = power_kw_sold * duration_hours
        
        battery_loss_kwh = grid_out_kwh / self.discharge_efficiency
        
        # Cap battery_loss_kwh to current energy
        battery_loss_kwh = min(battery_loss_kwh, self.soc_kwh)
        
        # Recompute grid_out_kwh = battery_loss_kwh * discharging efficiency 
        grid_out_kwh = battery_loss_kwh * self.discharge_efficiency
        
        self.soc -= battery_loss_kwh / self.capacity_kwh
        self.soc = min(max(self.soc, 0.0), 1.0)
        
        return grid_out_kwh
    
    def step(self, action, power_kw, duration_hours):
        # Update SOC based on action (charge/discharge) and power
        if action == 'charge':
            energy_bought = self.charge(power_kw_bought = power_kw, duration_hours = duration_hours)
            energy_sold = 0
        elif action == 'discharge':
            energy_sold = self.discharge(power_kw_sold = power_kw, duration_hours = duration_hours)
            energy_bought = 0
        elif action == 'hold':
            energy_bought, energy_sold = 0, 0
        else:
            raise ValueError("Action must be 'charge', 'discharge', or 'hold'.")
        return energy_bought, energy_sold, self.soc
    

# note to self: power is like the rate of energy transfer 
# and capacity is obviously just the max capacity the battery can store 