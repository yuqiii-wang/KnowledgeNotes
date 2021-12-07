package oop_example_java_code.service;

import java.util.ArrayList;

import oop_example_java_code.text.Note;
import oop_example_java_code.text.TextNote;
import oop_example_java_code.text.TextAndImageNote;

/*
    Store and Retriee Notes
*/
public class NoteStore implements NoteStoreInterface {

    // private ArrayList<Note> noteList = new ArrayList<>();
    private ArrayList<TextNote> noteTextList = new ArrayList<>();
    private ArrayList<TextAndImageNote> noteTextImgList = new ArrayList<>();

    public void storeNote(TextNote note){
        noteTextList.add(note);
    }
    public void storeNote(TextAndImageNote note){
        noteTextImgList.add(note);
    }
    // public void storeNote(Note note){
    //     noteList.add(note);
    // }

    // public void getAllNotes() {
    //     for (Note note : noteList){
    //         if (note instanceof TextNote){
    //             TextNote noteTx = (TextNote)note;
    //             System.out.println(noteTx);
    //         }
    //     }
    // }

    public void getAllTextNotes(){
        for (TextNote note : noteTextList){
            System.out.println(note);
        }
    }

    public void getAllTextAndImageNotes(){
        for (TextAndImageNote note : noteTextImgList){
            System.out.println(note);
        }
    }
    
}
