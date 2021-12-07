package oop_example_java_code.service;

import oop_example_java_code.text.Note;
import oop_example_java_code.text.TextNote;
import oop_example_java_code.text.TextAndImageNote;

public interface NoteStoreInterface {

    public void storeNote(TextNote note);

    public void storeNote(TextAndImageNote note);

    // public void storeNote(Note note);

    // public void getAllNotes();

    public void getAllTextNotes();

    public void getAllTextAndImageNotes();
}
