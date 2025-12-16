import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import os
from functools import wraps

def step_log(func):
    """Decorator to log every analysis step to logs.txt"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] Executing: {func.__name__}"
        print(f"📝 {log_msg}")
        
        with open("logs.txt", "a") as log_file:
            log_file.write(log_msg + "\n")
        
        result = func(*args, **kwargs)
        
        completion_msg = f"[{timestamp}] Completed: {func.__name__}"
        with open("logs.txt", "a") as log_file:
            log_file.write(completion_msg + "\n")
        
        return result
    return wrapper

def generate_dummy_csv(filename="flights.csv", num_records=1200):
    """Generate dummy airline delay dataset for testing"""
    print(f"🔧 Generating dummy dataset: {filename} with {num_records} records...")
    
    airlines = ["Air India", "IndiGo", "Vistara", "SpiceJet", "GoAir"]
    airports = ["DEL", "BOM", "BLR", "HYD", "MAA", "CCU", "AMD", "PNQ"]
    statuses = ["on-time", "delayed", "cancelled"]
    
    data = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(num_records):
        flight_id = f"{random.choice(['AI', '6E', 'UK', 'SG', 'G8'])}{random.randint(100, 999)}"
        airline = random.choice(airlines)
        origin = random.choice(airports)
        destination = random.choice([a for a in airports if a != origin])
        
        departure_hour = random.randint(5, 22)
        departure_minute = random.choice([0, 15, 30, 45])
        departure_time = f"{departure_hour:02d}:{departure_minute:02d}"
        
        delay_minutes = random.choices(
            [0, random.randint(5, 30), random.randint(31, 90), random.randint(91, 180)],
            weights=[0.6, 0.25, 0.10, 0.05]
        )[0]
        
        arrival_hour = (departure_hour + 2 + delay_minutes // 60) % 24
        arrival_minute = (departure_minute + delay_minutes % 60) % 60
        arrival_time = f"{arrival_hour:02d}:{arrival_minute:02d}"
        
        if delay_minutes == 0:
            status = "on-time"
        elif delay_minutes > 120:
            status = random.choice(["delayed", "cancelled"])
        else:
            status = "delayed"
        
        data.append({
            "flight_id": flight_id,
            "airline": airline,
            "origin": origin,
            "destination": destination,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "delay_minutes": delay_minutes,
            "status": status
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ Successfully generated {filename} with {num_records} records!\n")
    return filename

class FlightRecord:
    """Represents a single flight record"""
    
    def __init__(self, flight_id, airline, origin, destination, departure, arrival, delay, status):
        self.flight_id = flight_id
        self.airline = airline
        self.origin = origin
        self.destination = destination
        self.departure = departure
        self.arrival = arrival
        self.delay = delay
        self.status = status
    
    def to_dict(self):
        """Convert flight record to dictionary"""
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
        """Check if flight is delayed"""
        return self.delay > 0
    
    def delay_category(self):
        """Categorize delay level"""
        if self.delay == 0:
            return "On-time"
        elif self.delay <= 30:
            return "Moderate Delay"
        else:
            return "High Delay"
    
    def __repr__(self):
        return f"FlightRecord({self.flight_id}, {self.airline}, {self.delay}min)"

class AirlineAnalytics:
    """Main analytics engine for airline delay analysis"""
    
    def __init__(self, data_file="flights.csv", log_file="logs.txt"):
        self.data_file = data_file
        self.log_file = log_file
        self.df = None
        
        with open(self.log_file, "w") as f:
            f.write(f"=== Airline Analytics Log Started at {datetime.now()} ===\n")
    
    @step_log
    def load_data(self):
        """Load dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Successfully loaded {len(self.df)} records from {self.data_file}")
            print(f"📊 Columns: {list(self.df.columns)}\n")
            return True
        except FileNotFoundError:
            print(f"❌ Error: File {self.data_file} not found!")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    @step_log
    def clean_data(self):
        """Clean and preprocess data"""
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return False
        
        print("🧹 Cleaning data...")
        initial_count = len(self.df)
        
        self.df.drop_duplicates(subset=['flight_id'], inplace=True)
        
        self.df.dropna(subset=['flight_id', 'airline', 'delay_minutes'], inplace=True)
        
        self.df['delay_minutes'] = pd.to_numeric(self.df['delay_minutes'], errors='coerce')
        self.df.dropna(subset=['delay_minutes'], inplace=True)
        
        self.df = self.df[self.df['delay_minutes'] >= 0]
        
        final_count = len(self.df)
        removed = initial_count - final_count
        
        print(f"✅ Data cleaned: {removed} records removed, {final_count} records remaining\n")
        return True
    
    @step_log
    def filter_data(self, airline=None, airport=None, month=None):
        """Filter data by airline, airport, or month"""
        if self.df is None:
            print("❌ No data loaded.")
            return None
        
        filtered_df = self.df.copy()
        
        if airline:
            filtered_df = filtered_df[filtered_df['airline'] == airline]
            print(f"🔍 Filtered by airline: {airline}")
        
        if airport:
            filtered_df = filtered_df[
                (filtered_df['origin'] == airport) | 
                (filtered_df['destination'] == airport)
            ]
            print(f"🔍 Filtered by airport: {airport}")
        
        print(f"📊 Filtered results: {len(filtered_df)} records\n")
        return filtered_df
    
    @step_log
    def average_delay_by_airline(self):
        """Calculate average delay per airline"""
        if self.df is None:
            print("❌ No data loaded.")
            return None
        
        avg_delays = self.df.groupby('airline')['delay_minutes'].mean().sort_values(ascending=False)
        
        print("\n📊 Average Delay by Airline:")
        print("=" * 40)
        for airline, delay in avg_delays.items():
            print(f"{airline:20s} — {delay:.2f} mins")
        print("=" * 40 + "\n")
        
        return avg_delays
    
    @step_log
    def delay_trend(self):
        """Analyze delay trends"""
        if self.df is None:
            print("❌ No data loaded.")
            return None
        
        trend_data = self.df.groupby('airline').agg({
            'delay_minutes': ['mean', 'median', 'std', 'count']
        }).round(2)
        
        print("\n📈 Delay Trend Analysis:")
        print(trend_data)
        print()
        
        return trend_data
    
    @step_log
    def plot_delay_distribution(self, save=True):
        """Generate histogram of delay distribution"""
        if self.df is None:
            print("❌ No data loaded.")
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
            print(f"✅ Plot saved: {filename}\n")
        
        plt.show()
    
    @step_log
    def plot_average_delay_bar(self, save=True):
        """Generate bar chart of average delay by airline"""
        if self.df is None:
            print("❌ No data loaded.")
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
            print(f"✅ Plot saved: {filename}\n")
        
        plt.show()
    
    @step_log
    def plot_heatmap(self, save=True):
        """Generate airport vs delay heatmap"""
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        pivot_origin = self.df.groupby(['origin', 'airline'])['delay_minutes'].mean().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_origin, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Average Delay (minutes)'})
        plt.xlabel('Airline', fontsize=12, fontweight='bold')
        plt.ylabel('Origin Airport', fontsize=12, fontweight='bold')
        plt.title('Airport vs Airline Delay Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filename = 'airport_delay_heatmap.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Heatmap saved: {filename}\n")
        
        plt.show()
    
    @step_log
    def export_stats(self, filename="delay_summary.csv"):
        """Export statistics to CSV"""
        if self.df is None:
            print("❌ No data loaded.")
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
        
        print(f"✅ Statistics exported to {filename}")
        print(f"📊 Summary:\n{stats_df}\n")
        return True
    
    def show_summary(self):
        """Display data summary"""
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        print("\n" + "=" * 60)
        print("📊 DATASET SUMMARY")
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
    """Display the CLI menu"""
    print("\n" + "=" * 60)
    print("✈️  AIRLINE DELAY VISUAL ANALYTICS TOOL")
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
    """Main CLI application"""
    print("\n🚀 Welcome to Airline Delay Visual Analytics Tool!")
    print("ITM Skills University - Case Study 103\n")
    
    csv_file = "flights.csv"
    if not os.path.exists(csv_file):
        print(f"📂 Dataset not found. Generating dummy data...")
        generate_dummy_csv(csv_file, num_records=1200)
    
    analytics = AirlineAnalytics(data_file=csv_file)
    
    while True:
        display_menu()
        choice = input("👉 Enter your choice (0-9): ").strip()
        
        if choice == '1':
            analytics.load_data()
            analytics.clean_data()
        
        elif choice == '2':
            analytics.show_summary()
        
        elif choice == '3':
            analytics.average_delay_by_airline()
        
        elif choice == '4':
            print("\n🎨 Generating all plots...")
            analytics.plot_average_delay_bar()
            analytics.plot_delay_distribution()
            analytics.plot_heatmap()
            print("✅ All plots generated successfully!\n")
        
        elif choice == '5':
            analytics.plot_average_delay_bar()
        
        elif choice == '6':
            analytics.plot_delay_distribution()
        
        elif choice == '7':
            analytics.plot_heatmap()
        
        elif choice == '8':
            analytics.export_stats()
        
        elif choice == '9':
            print("\n🔍 Filter Options:")
            airline = input("Enter airline (or press Enter to skip): ").strip() or None
            airport = input("Enter airport code (or press Enter to skip): ").strip() or None
            filtered_df = analytics.filter_data(airline=airline, airport=airport)
            if filtered_df is not None:
                print(filtered_df.head(10))
        
        elif choice == '0':
            print("\n👋 Thank you for using Airline Delay Analytics Tool!")
            print("📝 Check logs.txt for execution history.")
            print("🎓 ITM Skills University - B.Tech CSE\n")
            break
        
        else:
            print("❌ Invalid choice! Please select 0-9.")
        
        input("\n⏎ Press Enter to continue...")

if __name__ == "__main__":
    main()