try:
    import os
    import json
    from train import LinearRegression
except ImportError as e:
    print(f"Error: Missing dependency. {e}")
    print("Please run: pip install pandas numpy matplotlib")
    sys.exit(1)


def main():
    while True:
        try:
            mileage = float(input("Please input the mileage: "))
            if mileage < 0:
                print("Error: Mileage cannot be negative.")
                continue
            break
        except ValueError:
            print(f"Error: Not a valid number. Try again.")
        except EOFError: # Ctrl+D
            exit(0)

    linearReg = LinearRegression()
    
    linearReg.load_model("model.json")

    estimate_price = linearReg.estimatePrice(mileage)
    print("estimated price", estimate_price)

if __name__ == "__main__":
    main()
