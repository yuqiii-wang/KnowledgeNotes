package design_pattern_java_code.chain_of_responsibility;

public class FirstLevelProcess extends Process {

    public FirstLevelProcess(int level){
        super.setLevel(level);
    }

    @Override
    public void illustrateProcess() {
       System.out.println("This is the first process.");
    }
}
