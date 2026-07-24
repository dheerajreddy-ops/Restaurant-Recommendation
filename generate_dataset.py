import csv
import random

random.seed(42)

cities = [
    "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
]

cuisine_pool = [
    "North Indian", "South Indian", "Chinese", "Italian", "Mexican",
    "Japanese", "Thai", "Mughlai", "Biryani", "Continental",
    "Street Food", "Desserts", "Bakery", "Kerala", "Bengali",
    "Punjabi", "Rajasthani", "Goan", "Seafood", "Vegan"
]

rest_types = [
    "Quick Bites", "Casual Dining", "Fine Dining", "Cafe",
    "Bar", "Pub", "Buffet", "Dessert Parlor", "Food Truck",
    "Sweet Shop", "Bakery", "Beverage Shop"
]

restaurant_names_pool = [
    "The Royal Kitchen", "Spice Garden", "Saffron Tales", "Curry House",
    "Tandoori Nights", "The Chai Point", "Blue Plate Cafe", "Masala Junction",
    "Biryani Blues", "Pizza Paradise", "Sushi Station", "Taco Fiesta",
    "Noodle Bar", "The Grill Room", "Burger Barn", "Pasta Palace",
    "Dosa Factory", "Idli Express", "Paneer Palace", "BBQ Nation",
    "Cafe Mocha", "The Chocolate Room", "Cravings", "Food Village",
    "Royal Thali", "Delhi Dhaba", "Mumbai Street", "Chennai Express",
    "Kolkata Kitchen", "Hyderabad House", "Pune Bites", "Jaipur Masala",
    "Gujarat Thali", "Lucknowi Kebab", "Goa Vibes", "Kerala Kitchen",
    "Bengali Bhoj", "Rajasthani Rasoi", "Mughal Garden", "Chinese Wok",
    "Thai Basil", "Japanese Zen", "Italiano", "Le French Corner",
    "Mexican Grill", "Turkish Delight", "Arabian Nights", "Persian Dhaba",
    "The Green Bowl", "Farm Fresh Cafe", "Organic Kitchen", "健康厨房",
    "The Food Co.", "Urban Diner", "Highway Dhaba", "Moonlight Cafe",
    "Sunrise Restaurant", "Sunset Grill", "Rainbow Cafe", "The Food Court",
    "Foodies Delight", "Yummy Tummy", "Taste of Home", "Grandmas Kitchen",
    "Chef's Table", "The Spice Route", "Coastal Kitchen", "Mountain Dhaba",
    "Garden View", "Rooftop Dining", "The Bistro", "Corner Cafe",
    "Street Side", "Chai & Snacks", "Midnight Cafe", "Early Bird",
    "The Brunch Club", "Dinner Club", "Family Dhaba", "Friends Cafe",
    "Bachelor's Kitchen", "College Canteen", "Office Dabba", "Senior Citizens",
    "Kids Zone Cafe", "Veg Paradise", "Non-Veg Heaven", "Mixed Grill",
    "Fried Chicken Hub", "Waffle House", "Pancake Kitchen", "Ice Cream Parlor",
    "Juice Bar", "Smoothie Station", "Coffee House", "Tea Lounge",
    "Brewery", "Wine Bar", "Cocktail Lounge", "Rooftop Bar",
]

def generate_dataset():
    rows = []
    used_names = set()

    for i in range(200):
        city = random.choice(cities)

        base_names = random.sample(restaurant_names_pool, 1)
        name = base_names[0]
        suffix = random.choice(["", "", "", "", f" - {city}", f" {chr(65 + i % 26)}"])
        name = name + suffix
        if name in used_names:
            name = name + f" {i}"
        used_names.add(name)

        online_order = random.choice(["Yes", "No", "Yes", "Yes"])
        book_table = random.choice(["Yes", "No", "No", "No"])

        rate = round(random.uniform(2.5, 4.9), 1)
        if rate > 4.5:
            rate = round(random.uniform(4.2, 4.8), 1)

        votes = random.randint(5, 8500)

        location_parts = random.sample(["Main Road", "Cross Road", "Market Area",
            "Mall", "Complex", "Tower", "Street", "Lane", "Nagar", "Park",
            "Garden", "Plaza", "Tower", "Hills", "Lake View", "Station Road",
            "Highway", "Circle", "Square", "Extension"], k=random.randint(1, 2))
        location = city + ", " + " ".join(location_parts)

        rest_type = random.choice(rest_types)

        num_cuisines = random.randint(1, 4)
        selected_cuisines = random.sample(cuisine_pool, num_cuisines)
        cuisines = ", ".join(selected_cuisines)

        cost_bracket = random.choice([1, 2, 3, 4, 5, 6])
        cost_map = {1: (100, 300), 2: (200, 500), 3: (300, 800),
                    4: (500, 1200), 5: (800, 2000), 6: (1500, 3000)}
        lo, hi = cost_map[cost_bracket]
        approx_cost_for_two = random.randint(lo, hi)

        rows.append({
            "name": name,
            "online_order": online_order,
            "book_table": book_table,
            "rate": rate,
            "votes": votes,
            "location": location,
            "rest_type": rest_type,
            "cuisines": cuisines,
            "approx_cost_for_two": approx_cost_for_two,
            "location_encoded": 0,
            "rest_type_encoded": 0,
            "cuisines_encoded": 0
        })

    header = ["name", "online_order", "book_table", "rate", "votes",
              "location", "rest_type", "cuisines", "approx_cost_for_two",
              "location_encoded", "rest_type_encoded", "cuisines_encoded"]

    with open("dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} restaurant records in dataset.csv")

if __name__ == "__main__":
    generate_dataset()
