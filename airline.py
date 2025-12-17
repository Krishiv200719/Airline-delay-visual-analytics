import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def step_log(func):
    def wrapper(*args, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] Executing: {func.__name__}"
        print(log_msg)
        
        with open("logs.txt", "a") as log_file:
            log_file.write(log_msg + "\n")
        
        result = func(*args, **kwargs)
        
        completion_msg = f"[{timestamp}] Completed: {func.__name__}"
        with open("logs.txt", "a") as log_file:
            log_file.write(completion_msg + "\n")
        
        return result
    return wrapper

class FlightRecord:
    def __init__(self, flight_id, airline, origin, destination, 
                 departure, arrival, delay, status):
        self.flight_id = flight_id
        self.airline = airline
        self.origin = origin
        self.destination = destination
        self.departure = departure
        self.arrival = arrival
        self.delay = delay
        self.status = status
    
    def to_dict(self):
        return {
            "flight_id": self.flight_id,
            "airline": self.airline,
            "origin": self.origin,
            "destination": self.destination,
            "departure": self.departure,
            "arrival": self.arrival,
            "delay": self.delay,
            "status": self.status
        }
    
    def is_delayed(self):
        return self.delay > 0
    
    def delay_category(self):
        if self.delay == 0:
            return "On-time"
        elif self.delay <= 30:
            return "Moderate Delay"
        else:
            return "High Delay"
    
    def __repr__(self):
        return f"FlightRecord({self.flight_id}, {self.airline}, {self.delay}min)"

class AirlineAnalytics:
    def __init__(self, data_file="flights.csv", log_file="logs.txt"):
        self.data_file = data_file
        self.log_file = log_file
        self.df = None
        
        with open(self.log_file, "w") as f:
            f.write(f"=== Airline Analytics Log Started at {datetime.now()} ===\n")
    
    @step_log
    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"Successfully loaded {len(self.df)} records from {self.data_file}")
            print(f"Columns: {list(self.df.columns)}\n")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.data_file} not found!")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    @step_log
    def filter_data(self, airline=None, airport=None, month=None):
        if self.df is None:
            print("No data loaded.")
            return None
        
        filtered_df = self.df.copy()
        
        if airline:
            filtered_df = filtered_df[filtered_df['airline'] == airline]
            print(f"Filtered by airline: {airline}")
        
        if airport:
            filtered_df = filtered_df[
                (filtered_df['origin'] == airport) | 
                (filtered_df['destination'] == airport)
            ]
            print(f"Filtered by airport: {airport}")
        
        print(f"Filtered results: {len(filtered_df)} records\n")
        return filtered_df
    
    @step_log
    def average_delay_by_airline(self):
        if self.df is None:
            print("No data loaded.")
            return None
        
        avg_delays = self.df.groupby('airline')['delay_minutes'].mean().sort_values(ascending=False)
        
        print("\nAverage Delay by Airline:")
        print("=" * 40)
        for airline, delay in avg_delays.items():
            print(f"{airline:20s} - {delay:.2f} mins")
        print("=" * 40 + "\n")
        
        return avg_delays
    
    @step_log
    def delay_trend(self):
        if self.df is None:
            print("No data loaded.")
            return None
        
        trend_data = self.df.groupby('airline').agg({
            'delay_minutes': ['mean', 'median', 'std', 'count']
        }).round(2)
        
        print("\nDelay Trend Analysis:")
        print(trend_data)
        print()
        
        return trend_data
    
    @step_log
    def plot_delay_distribution(self, save=True):
        if self.df is None:
            print("No data loaded.")
            return
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['delay_minutes'], bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Delay (minutes)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Overall Delay Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for airline in self.df['airline'].unique():
            airline_data = self.df[self.df['airline'] == airline]['delay_minutes']
            plt.hist(airline_data, alpha=0.5, label=airline, bins=30)
        
        plt.xlabel('Delay (minutes)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Delay Distribution by Airline', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = 'delay_distribution.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {filename}\n")
        
        plt.show()
    
    @step_log
    def plot_average_delay_bar(self, save=True):
        if self.df is None:
            print("No data loaded.")
            return
        
        avg_delays = self.df.groupby('airline')['delay_minutes'].mean().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(avg_delays.index, avg_delays.values, color='coral', edgecolor='black')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Airline', fontsize=12, fontweight='bold')
        plt.ylabel('Average Delay (minutes)', fontsize=12, fontweight='bold')
        plt.title('Average Delay by Airline', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = 'delay_by_airline.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {filename}\n")
        
        plt.show()
    
    @step_log
    def plot_heatmap(self, save=True):
        if self.df is None:
            print("No data loaded.")
            return
        
        pivot_origin = self.df.groupby(['origin', 'airline'])['delay_minutes'].mean().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(pivot_origin.values, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(pivot_origin.columns)))
        ax.set_yticks(np.arange(len(pivot_origin.index)))
        ax.set_xticklabels(pivot_origin.columns)
        ax.set_yticklabels(pivot_origin.index)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(len(pivot_origin.index)):
            for j in range(len(pivot_origin.columns)):
                text = ax.text(j, i, f'{pivot_origin.values[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_xlabel('Airline', fontsize=12, fontweight='bold')
        ax.set_ylabel('Origin Airport', fontsize=12, fontweight='bold')
        ax.set_title('Airport vs Airline Delay Heatmap', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Delay (minutes)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save:
            filename = 'airport_delay_heatmap.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved: {filename}\n")
        
        plt.show()
    
    @step_log
    def export_stats(self, filename="delay_summary.csv"):
        if self.df is None:
            print("No data loaded.")
            return False
        
        stats = []
        
        for airline in self.df['airline'].unique():
            airline_data = self.df[self.df['airline'] == airline]
            stats.append({
                'airline': airline,
                'total_flights': len(airline_data),
                'avg_delay': airline_data['delay_minutes'].mean(),
                'median_delay': airline_data['delay_minutes'].median(),
                'max_delay': airline_data['delay_minutes'].max(),
                'min_delay': airline_data['delay_minutes'].min(),
                'std_delay': airline_data['delay_minutes'].std(),
                'on_time_pct': (airline_data['delay_minutes'] == 0).sum() / len(airline_data) * 100,
                'delayed_pct': (airline_data['delay_minutes'] > 0).sum() / len(airline_data) * 100
            })
        
        stats_df = pd.DataFrame(stats)
        stats_df = stats_df.round(2)
        stats_df.to_csv(filename, index=False)
        
        print(f"Statistics exported to {filename}")
        print(f"Summary:\n{stats_df}\n")
        return True
    
    def show_summary(self):
        if self.df is None:
            print("No data loaded.")
            return
        
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total Records: {len(self.df)}")
        print(f"Airlines: {', '.join(self.df['airline'].unique())}")
        print(f"Airports: {', '.join(self.df['origin'].unique())}")
        print(f"\nDelay Statistics:")
        print(f"  Average Delay: {self.df['delay_minutes'].mean():.2f} minutes")
        print(f"  Median Delay: {self.df['delay_minutes'].median():.2f} minutes")
        print(f"  Max Delay: {self.df['delay_minutes'].max():.2f} minutes")
        print(f"  On-time Flights: {(self.df['delay_minutes'] == 0).sum()} ({(self.df['delay_minutes'] == 0).sum() / len(self.df) * 100:.1f}%)")
        print(f"  Delayed Flights: {(self.df['delay_minutes'] > 0).sum()} ({(self.df['delay_minutes'] > 0).sum() / len(self.df) * 100:.1f}%)")
        print("=" * 60 + "\n")

def display_menu():
    print("\n" + "=" * 60)
    print("AIRLINE DELAY VISUAL ANALYTICS TOOL")
    print("=" * 60)
    print("1. Load Dataset")
    print("2. Show Summary")
    print("3. Average Delay by Airline")
    print("4. Generate All Plots")
    print("5. Generate Bar Chart (Avg Delay)")
    print("6. Generate Histogram (Delay Distribution)")
    print("7. Generate Heatmap (Airport vs Delay)")
    print("8. Export Statistics to CSV")
    print("9. Filter Data")
    print("0. Exit")
    print("=" * 60)

def main():
    csv_file = "flights.csv"
    analytics = AirlineAnalytics(data_file=csv_file)
    
    while True:
        display_menu()
        choice = input("Enter your choice (0-9): ").strip()
        
        if choice == '1':
            analytics.load_data()
        
        elif choice == '2':
            analytics.show_summary()
        
        elif choice == '3':
            analytics.average_delay_by_airline()
        
        elif choice == '4':
            print("\nGenerating all plots...")
            analytics.plot_average_delay_bar()
            analytics.plot_delay_distribution()
            analytics.plot_heatmap()
            print("All plots generated successfully!\n")
        
        elif choice == '5':
            analytics.plot_average_delay_bar()
        
        elif choice == '6':
            analytics.plot_delay_distribution()
        
        elif choice == '7':
            analytics.plot_heatmap()
        
        elif choice == '8':
            analytics.export_stats()
        
        elif choice == '9':
            print("\nFilter Options:")
            airline = input("Enter airline (or press Enter to skip): ").strip() or None
            airport = input("Enter airport code (or press Enter to skip): ").strip() or None
            filtered_df = analytics.filter_data(airline=airline, airport=airport)
            if filtered_df is not None:
                print(filtered_df.head(10))
        
        elif choice == '0':
            print("\nThank you for using Airline Delay Analytics Tool!")
            print("Check logs.txt for execution history.")
            print("ITM Skills University - B.Tech CSE\n")
            break
        
        else:
            print("Invalid choice! Please select 0-9.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()