package design_pattern_java_code.factory;

public class DomesticPlan extends Plan {

    @Override
    public void setRate(double newRate) {
        super.rate = newRate;
    }

    @Override
    public double getRate() {
        return super.rate;
    }
}
