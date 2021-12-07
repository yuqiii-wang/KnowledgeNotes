package design_pattern_java_code.factory;

public class ForeignPlan extends Plan {
    @Override
    public void setRate(double newRate) {
        super.rate = newRate * 2;
    }

    @Override
    public double getRate() {
        return super.rate;
    }
}
