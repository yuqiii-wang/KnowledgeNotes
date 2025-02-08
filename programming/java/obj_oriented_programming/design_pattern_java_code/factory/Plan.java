package design_pattern_java_code.factory;

public abstract class Plan {
    protected double rate;

    public abstract double getRate();

    public abstract void setRate(double newRate);

    public void calculateBill(int units){
        System.out.println("Result: $" + Double.toString(units * rate));
    }
}
