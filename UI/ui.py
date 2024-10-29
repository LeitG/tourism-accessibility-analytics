import pickle
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QListWidget, QPushButton, QMessageBox, QDialog, QTextEdit, QLabel

class CommentDialog(QDialog):
    def __init__(self, review, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Full Review')
        self.setGeometry(parent.geometry().center().x() - 200, parent.geometry().center().y() - 100, 400, 200)

        layout = QVBoxLayout()
        self.reviewLabel = QTextEdit(review)
        self.reviewLabel.setReadOnly(True)
        layout.addWidget(self.reviewLabel)

        self.setLayout(layout)

class RecommendationDialog(QDialog):
    def __init__(self, hotel, reason, comments, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Hotel Recommendation')
        self.setGeometry(parent.geometry().center().x() - 200, parent.geometry().center().y() - 100, 400, 400)

        layout = QVBoxLayout()

        self.infoLabel = QTextEdit(f'The recommended hotel is: {hotel}<br>Reason: {reason}')
        self.infoLabel.setReadOnly(True)
        layout.addWidget(self.infoLabel)

        self.reviewLabel = QLabel('Reviews from other users:')
        layout.addWidget(self.reviewLabel)

        self.reviewListWidget = QListWidget()
        self.reviews = comments
        for review in self.reviews:
            truncated_review = review[:20]
            if len(review) > 20:
                self.reviewListWidget.addItem("Review: " + truncated_review + "......")
            else:
                self.reviewListWidget.addItem("Review: " + truncated_review)
            
        self.reviewListWidget.itemDoubleClicked.connect(self.show_full_review)
        layout.addWidget(self.reviewListWidget)

        self.setLayout(layout)

    def show_full_review(self, item):
        index = self.reviewListWidget.row(item)
        full_review = self.reviews[index]
        dialog = CommentDialog(full_review, self)
        dialog.exec_()


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Accessible Hotel Recommendations')
        self.setGeometry(100, 100, 600, 400)

        self.layout = QVBoxLayout()

        # Create user list
        self.userListWidget = QListWidget()
        self.users = pickle.load(open('UI.pickle', 'rb'))
        self.userListWidget.addItems(self.users.keys())
        self.layout.addWidget(self.userListWidget)

        # Create button
        self.button = QPushButton('Show Recommended Hotel')
        self.button.setEnabled(False)
        self.button.clicked.connect(self.on_button_click)
        self.layout.addWidget(self.button)

        self.userListWidget.itemDoubleClicked.connect(self.show_reviews)

        self.setLayout(self.layout)

    def on_button_click(self):
        selected_user = self.userListWidget.property("selected_user")
        
        hotel = str(self.users[selected_user]['target'])
        
        reason = self.users[selected_user]['true']
        comments = self.users[selected_user]['other_reviews']
        dialog = RecommendationDialog(hotel, reason, comments, self)
        dialog.exec_()

    def show_reviews(self, item):
        selected_user = item.text()
        reviews = self.users[selected_user]['user_reviews']
        self.userListWidget.clear()
        for review in reviews:
            truncated_review = review[:40]
            if len(review) > 20:
                self.userListWidget.addItem("Review: " + truncated_review + "......")
            else:
                self.userListWidget.addItem("Review: " + truncated_review)
        self.userListWidget.setProperty("full_reviews", reviews)
        self.userListWidget.setProperty("selected_user", selected_user)
        self.button.setEnabled(True)
        self.userListWidget.itemDoubleClicked.disconnect(self.show_reviews)
        self.userListWidget.itemDoubleClicked.connect(self.show_full_review)

    def show_full_review(self, item):
        full_reviews = self.userListWidget.property("full_reviews")
        if full_reviews:
            index = self.userListWidget.row(item)
            full_review = full_reviews[index]
            dialog = CommentDialog(full_review, self)
            dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = AppDemo()
    demo.show()
    sys.exit(app.exec_())
