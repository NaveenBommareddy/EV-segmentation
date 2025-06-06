# EV sales
import matplotlib.pyplot as plt

vehicle_types = ['Two-Wheelers (E2W)', 'Three-Wheelers (E3W)', 'Passenger Cars', 'Buses & Commercial']
sales_distribution = [60, 25, 10, 5]

plt.figure(figsize=(6,6))
plt.pie(sales_distribution, labels=vehicle_types, autopct='%1.1f%%', startangle=140)
plt.title('EV Sales Distribution by Vehicle Type (2024)')
plt.show()


# EV market size growth
years = [2020, 2021, 2022, 2023, 2024, 2030]
market_size = [5000, 8000, 14000, 23000, 38000, 300000]  # in ₹ Crores

plt.figure(figsize=(8,5))
plt.plot(years, market_size, marker='o', linestyle='-', color='teal')
plt.title('EV Market Size Growth in India (₹ Cr)')
plt.xlabel('Year')
plt.ylabel('Market Size (₹ Cr)')
plt.grid(True)
plt.show()


#cost comparision petrol vs EV
import numpy as np

categories = ['Initial Cost', 'Running Cost (5 yrs)', 'Total Cost']
petrol = [6, 5, 11]  # ₹ Lakhs
ev = [9, 1.5, 10.5]  # ₹ Lakhs

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, petrol, width, label='Petrol Car', color='coral')
plt.bar(x + width/2, ev, width, label='EV Car', color='mediumseagreen')
plt.xticks(x, categories)
plt.ylabel('Cost (₹ Lakhs)')
plt.title('EV vs Petrol Car: 5-Year Cost Comparison')
plt.legend()
plt.show()


# challenges in EV adaption
import matplotlib.pyplot as plt
import numpy as np

labels = ['High Upfront Cost', 'Charging Infra', 'Battery Life', 'Awareness', 'Model Availability']
values = [4.5, 4, 3.5, 2.5, 3]
values += values[:1]  # close the loop

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.plot(angles, values, color='slateblue', linewidth=2)
ax.fill(angles, values, color='slateblue', alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticklabels([])
ax.set_title('Challenges in EV Adoption')
plt.show()
