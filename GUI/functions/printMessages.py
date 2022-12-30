from PyQt5.QtWidgets import QMessageBox


def printCritical(message):
    messageBox = QMessageBox()
    messageBox.setIcon(QMessageBox.Critical)
    # Print the message
    messageBox.setText(message)
    messageBox.setWindowTitle("Critical")
    # Ok button
    messageBox.setStandardButtons(QMessageBox.Ok)
    # Execute
    messageBox.exec_()
    return


def printAccepted(message):
    messageBox = QMessageBox()
    messageBox.setIcon(QMessageBox.Accepted)
    # Print the message
    messageBox.setText(message)
    messageBox.setWindowTitle("Accepted")
    # Ok button
    messageBox.setStandardButtons(QMessageBox.Ok)
    # Execute
    messageBox.exec_()
    return


def printRejected(message):
    messageBox = QMessageBox()
    messageBox.setIcon(QMessageBox.Rejected)
    # Print the message
    messageBox.setText(message)
    messageBox.setWindowTitle("Rejected")
    # Ok button
    messageBox.setStandardButtons(QMessageBox.Ok)
    # Execute
    messageBox.exec_()
    return


def printInfo(message):
    messageBox = QMessageBox()
    messageBox.setIcon(QMessageBox.Information)
    # Print the message
    messageBox.setText(message)
    messageBox.setWindowTitle("Information")
    # Ok button
    messageBox.setStandardButtons(QMessageBox.Ok)
    # Execute
    messageBox.exec_()
    return
