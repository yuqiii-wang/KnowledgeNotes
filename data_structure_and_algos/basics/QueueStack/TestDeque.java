/*
    The Java Deque interface, java.util.Deque, 
    represents a double ended queue, meaning 
    a queue where you can add and remove elements 
    to and from both ends of the queue. 
    The name Deque is an abbreviation of Double Ended Queue.
*/

import java.util.LinkedList;
import java.util.ArrayDeque;
import java.util.Deque;

public class TestDeque
{
    TestDeque(){}

    public void run()
    {
        Deque<Integer> dequeList = new LinkedList<Integer>();
        Deque<Integer> dequeArr = new ArrayDeque<Integer>();

        for (int i = 0; i < 10; i++)
        {
            dequeList.add(i);
            dequeArr.add(i);
        }

        for (Integer iter_int : dequeList)
        {
            System.out.println(iter_int);
        }

        for (Integer iter_int : dequeArr)
        {
            System.out.println(iter_int);
        }
        System.out.println();
    }

    public static void main(String args[])
    {
        TestDeque tstDeque = new TestDeque();
        tstDeque.run();
    }
}