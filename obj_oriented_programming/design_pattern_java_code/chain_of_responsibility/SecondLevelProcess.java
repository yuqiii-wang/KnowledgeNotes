package design_pattern_java_code.chain_of_responsibility;

public class SecondLevelProcess extends Process {

    public SecondLevelProcess(int level){
        super.setLevel(level);
    }

    @Override
    public void illustrateProcess() {
       System.out.println("This is the second process.");
    }
}
