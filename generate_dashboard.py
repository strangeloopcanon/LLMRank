#!/usr/bin/env python3
from dashboard import generate_html

if __name__ == "__main__":
    dashboard_path = generate_html()
    print(f"Dashboard HTML generated at {dashboard_path}")
    print("You can open this file in a web browser to view the dashboard.")