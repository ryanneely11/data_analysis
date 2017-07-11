## this file is a repository for Tkinter code that might be reused.

try:
	import tkinter as Tk
except ImportError:
	print("tkinter not imported")

##a function to create a multi-selection list box, and
###return a list of user-selected items.
def multichoice(choices, title = "Make one or more selection"):
	##container for chosen items
	chosen = []
	##create a Tkinter root window
	root = Tk.Tk()
	#initialize the listbox with multiple selections
	lb = Tk.Listbox(root, selectmode = 'multiple')
	##add choices to the listbox
	for item in choices:
		lb.insert('end', item)
	##create a title
	title = Tk.Label(root, text = title)
	##create a button function
	def callback():
		selection = lb.curselection()
		for index in selection:
			chosen.append(choices[int(index)])
		root.destroy()
	b = Tk.Button(root, text = "Select", command = callback)
	##create a scrollbar
	sb = Tk.Scrollbar(root, orient = "vertical")
	sb.config(command = lb.yview)
	sb.pack(side = "right", fill = "y")
	lb.config(yscrollcommand = sb.set)
	lb.pack()
	title.pack()
	b.pack()
	root.mainloop()
	return chosen

def onechoice(choices, title = "Make one selection"):
	##create a Tkinter root window
	root = Tk.Tk()
	##a list to store the result; I know, sort of awkward
	chosen = []
	#initialize the listbox with multiple selections
	lb = Tk.Listbox(root, selectmode = 'single')
	##add choices to the listbox
	for item in choices:
		lb.insert('end', item)
	##create a title
	title = Tk.Label(root, text = title)
	##create a button function
	def callback():
		selection = lb.curselection()
		chosen.append(choices[int(selection[0])])
		root.destroy()
	b = Tk.Button(root, text = "Select", command = callback)
	##create a scrollbar
	sb = Tk.Scrollbar(root, orient = "vertical")
	sb.config(command = lb.yview)
	sb.pack(side = "right", fill = "y")
	lb.config(yscrollcommand = sb.set)
	lb.pack()
	title.pack()
	b.pack()
	root.mainloop()
	return chosen[0]
